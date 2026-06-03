import argparse
import gc
import io
import json
import os
import random
from pathlib import Path
from typing import Mapping, Sequence, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from PIL import Image, ImageFile, PngImagePlugin
from datasets import load_dataset, concatenate_datasets, Image as HFImage
import tensorflow as tf


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

try:
    RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE = Image.BILINEAR

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

CAPTIONS: Sequence[Mapping[str, str]] = ()
FOLDER_PATHS: Optional[List[Path]] = None
FOLDER_ROOT: Optional[str] = None


def _init_worker(
    captions: Sequence[Mapping[str, str]],
    folder_paths: Optional[List[Path]],
    folder_root: Optional[str],
) -> None:
    global CAPTIONS, FOLDER_PATHS, FOLDER_ROOT
    CAPTIONS = captions
    FOLDER_PATHS = folder_paths
    FOLDER_ROOT = folder_root


def _iter_image_files(root: Path) -> List[Path]:
    paths: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(IMAGE_EXTS):
                paths.append(Path(dirpath) / fname)
    return sorted(paths)


def _get_caption_list(
    image_name: str,
    caption_sets: Sequence[Mapping[str, str]],
) -> Optional[List[str]]:
    captions: List[str] = []
    for caption_map in caption_sets:
        caption = caption_map.get(image_name)
        if caption is None:
            return None
        captions.append(caption if isinstance(caption, str) else str(caption))
    return captions


def _load_hf_caption_maps(
    subset: str,
    caption_columns: Sequence[str],
    num_proc: int,
) -> List[Mapping[str, str]]:
    ds = load_dataset(
        "zlab-princeton/i1-captions",
        subset,
        split="train",
        keep_in_memory=False,
        features=None,
        num_proc=num_proc,
    )
    missing_columns = [
        column
        for column in caption_columns
        if column not in ds.column_names
    ]
    if missing_columns:
        raise KeyError(
            f"Subset {subset} does not have columns: {missing_columns}"
        )

    caption_maps: List[dict[str, str]] = [dict() for _ in caption_columns]
    for row_idx, row in enumerate(ds):
        key = str(row["key"])
        for caption_map, column in zip(caption_maps, caption_columns):
            caption = row[column]
            if caption is not None:
                caption_map[key] = caption if isinstance(caption, str) else str(caption)
    return caption_maps


def _image_name_from_path(root: str, path: Path) -> str:
    if "places365-challenge2016" in root:
        return f"{path.parent.name}_{path.stem}"
    return path.stem


def _resize_shorter_edge_and_center_crop(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    width, height = img.size
    if width <= height:
        new_w = size
        new_h = int(round(height * (size / width)))
    else:
        new_h = size
        new_w = int(round(width * (size / height)))
    if (new_w, new_h) != (width, height):
        img = img.resize((new_w, new_h), resample=RESAMPLE)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _shorter_edge_too_small(img: Image.Image, size: int) -> bool:
    width, height = img.size
    return min(width, height) < size


def _open_image_from_source(image_source):
    if isinstance(image_source, Image.Image):
        return image_source
    if image_source is None:
        raise ValueError("Received empty image source.")
    if isinstance(image_source, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_source))
    if isinstance(image_source, str):
        return Image.open(image_source)
    if hasattr(image_source, "read"):
        return Image.open(image_source)
    if isinstance(image_source, dict):
        if "bytes" in image_source and isinstance(image_source["bytes"], (bytes, bytearray)):
            return Image.open(io.BytesIO(image_source["bytes"]))
        if "path" in image_source and isinstance(image_source["path"], str):
            return Image.open(image_source["path"])
    raise TypeError(f"Unsupported image source type: {type(image_source)}")


def _disable_hf_image_decoding(ds):
    for name, feature in ds.features.items():
        if isinstance(feature, HFImage):
            ds = ds.cast_column(name, HFImage(decode=False))
    return ds


def _load_hf_dataset(spec: str, num_proc: int):
    if spec == "imagenet22k":
        return load_dataset(
            "timm/imagenet-22k-wds",
            split="train",
            keep_in_memory=False,
            features=None,
            num_proc=num_proc,
        )
    if spec == "fluxreason":
        return load_dataset(
            "LucasFang/FLUX-Reason-6M",
            split="train",
            keep_in_memory=False,
            features=None,
            num_proc=num_proc,
        )
    if spec == "pexels":
        return load_dataset(
            "animetimm/pexels-tagger-v0-w640-ws-full",
            split="train",
            keep_in_memory=False,
            features=None,
            num_proc=num_proc,
        )
    if spec == "gptedit":
        return load_dataset(
            "UCSC-VLAA/gpt-edit-simpler",
            split="train",
            keep_in_memory=False,
            features=None,
            num_proc=num_proc,
        )
    if spec == "rendered_text":
        return load_dataset(
            "wendlerc/RenderedText",
            split="train",
            keep_in_memory=False,
            features=None,
            num_proc=num_proc,
        )
    if spec == "textatlas":
        parts = []
        subsets = [
            "CleanTextSynth",
            "PPT2Details",
            "PPT2Structured",
            "LongWordsSubset-A",
            "LongWordsSubset-M",
            "CoverBook",
            "Paper2Text",
            "TextVisionBlend",
            "StyledTextSynth",
            "TextScenesHQ",
        ]
        for subset in subsets:
            d = load_dataset(
                "CSU-JPG/TextAtlas5M",
                subset,
                split="train",
                keep_in_memory=False,
                features=None,
                num_proc=num_proc,
            )
            def add_global_id(ex, subset=subset):
                ex["global_id"] = f"{subset}:{ex['image_path']}"
                return ex
            d = d.map(add_global_id, num_proc=num_proc)
            parts.append(d)
        return concatenate_datasets(parts)
    return load_dataset(
        spec,
        split="train",
        keep_in_memory=False,
        features=None,
        num_proc=num_proc,
    )


def _extract_hf_name(entry) -> str:
    if "image_id" in entry:
        return str(entry["image_id"])
    if "__key__" in entry:
        return str(entry["__key__"])
    if "id" in entry:
        return str(entry["id"])
    if "global_id" in entry:
        return str(entry["global_id"])
    raise KeyError("No supported image id field in HF example.")


def _extract_hf_image_source(entry):
    image_source = entry.get("image")
    if image_source is None:
        if "jpg" in entry:
            image_source = entry["jpg"]
        elif "webp" in entry:
            image_source = entry["webp"]
        elif "output" in entry:
            image_source = entry["output"]
        elif "png" in entry:
            image_source = entry["png"]
    return image_source


def _split_indices(total: int, num_shards: int) -> List[Tuple[int, int]]:
    base = total // num_shards
    remainder = total % num_shards
    indices: List[Tuple[int, int]] = []
    start = 0
    for shard_idx in range(num_shards):
        size = base + (1 if shard_idx < remainder else 0)
        end = start + size
        indices.append((start, end))
        start = end
    return indices


def _load_shape_metadata(path: str | Path) -> dict[str, int]:
    metadata_path = Path(path)
    metadata: dict[str, int] = {}
    with open(metadata_path, "r") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_name = record.get("image_name")
            short_edge = record.get("short_edge")
            if image_name is None or short_edge is None:
                raise ValueError(f"Invalid shape metadata record at line {line_idx}: {line[:200]}")
            metadata[str(image_name)] = int(short_edge)
            if line_idx % 1000000 == 0:
                print(f"Loaded {line_idx} shape metadata records so far.", flush=True)
    print(f"Loaded shape metadata for {len(metadata)} images from {metadata_path}.", flush=True)
    return metadata


def _filter_folder_paths_by_shape_metadata(
    paths: Sequence[Path],
    root: str,
    shape_metadata: Mapping[str, int],
    image_size: int,
) -> Tuple[List[Path], int]:
    eligible_paths: List[Path] = []
    missing_metadata = 0
    total_paths = len(paths)
    for idx, path in enumerate(paths, start=1):
        short_edge = shape_metadata.get(_image_name_from_path(root, path))
        if short_edge is None:
            missing_metadata += 1
            continue
        if short_edge >= image_size:
            eligible_paths.append(path)
        if idx % 100000 == 0:
            print(
                f"Checked {idx}/{total_paths} folder images against shape metadata; "
                f"{len(eligible_paths)} eligible so far.",
                flush=True,
            )
    return eligible_paths, missing_metadata


def _filter_hf_indices_by_shape_metadata(ds, shape_metadata: Mapping[str, int], image_size: int) -> Tuple[List[int], int]:
    eligible_indices: List[int] = []
    missing_metadata = 0
    ds_len = len(ds)
    for idx in range(ds_len):
        try:
            image_name = _extract_hf_name(ds[idx])
        except Exception:
            missing_metadata += 1
            continue
        short_edge = shape_metadata.get(image_name)
        if short_edge is None:
            missing_metadata += 1
            continue
        if short_edge >= image_size:
            eligible_indices.append(idx)
        if (idx + 1) % 100000 == 0:
            print(
                f"Checked {idx + 1}/{ds_len} dataset images against shape metadata; "
                f"{len(eligible_indices)} eligible so far.",
                flush=True,
            )
    return eligible_indices, missing_metadata


def _reservoir_consider(sampled_items: List, item, total: int, limit: int) -> None:
    if limit <= 0:
        return
    if len(sampled_items) < limit:
        sampled_items.append(item)
    else:
        replacement_idx = random.randrange(total)
        if replacement_idx < limit:
            sampled_items[replacement_idx] = item


def _sample_folder_paths(paths: Sequence[Path], limit: int, image_size: int) -> Tuple[List[Path], int]:
    sampled_paths: List[Path] = []
    total = 0
    num_paths = len(paths)
    for idx, path in enumerate(paths, start=1):
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                if _shorter_edge_too_small(img, image_size):
                    continue
        except Exception:
            continue
        total += 1
        _reservoir_consider(sampled_paths, path, total, limit)
        if idx % 100000 == 0:
            print(
                f"Scanned {idx}/{num_paths} folder images for shorter-edge filtering; {total} eligible so far.",
                flush=True,
            )
    return sampled_paths, total


def _sample_hf_indices(ds, limit: int, image_size: int) -> Tuple[List[int], int]:
    sampled_indices: List[int] = []
    total = 0
    ds_len = len(ds)
    for idx in range(ds_len):
        try:
            entry = ds[idx]
            image_source = _extract_hf_image_source(entry)
            img = _open_image_from_source(image_source)
            if _shorter_edge_too_small(img, image_size):
                continue
        except Exception:
            continue
        total += 1
        _reservoir_consider(sampled_indices, idx, total, limit)
        if (idx + 1) % 100000 == 0:
            print(
                f"Scanned {idx + 1}/{ds_len} dataset images for shorter-edge filtering; {total} eligible so far.",
                flush=True,
            )
    return sampled_indices, total


def _write_tfrecord_shard(
    shard_idx: int,
    shard_tasks: Sequence[Tuple],
    output_dir: Path,
    total_shards: int,
    image_size: int,
    filter_shorter_edge: bool,
) -> Tuple[int, int, int]:
    FeatureMsg = tf.train.Feature
    BytesListMsg = tf.train.BytesList
    Int64ListMsg = tf.train.Int64List
    ExampleMsg = tf.train.Example
    FeaturesMsg = tf.train.Features

    tfrecord_path = output_dir / f"dataset-train.tfrecord-{shard_idx:05d}-of-{total_shards:05d}"
    writer = tf.io.TFRecordWriter(str(tfrecord_path))
    write = writer.write
    written = 0
    missing = 0
    decode_errors = 0
    filtered = 0
    hf_cache: dict[str, object] = {}

    try:
        for task in shard_tasks:
            kind = task[0]
            if kind == "folder":
                start, end = task[1], task[2]
                paths = FOLDER_PATHS or []
                root = FOLDER_ROOT or ""
                captions = CAPTIONS
                for path in paths[start:end]:
                    image_name = _image_name_from_path(root, path)
                    caption_list = _get_caption_list(image_name, captions)
                    if caption_list is None:
                        missing += 1
                        continue
                    try:
                        with open(path, "rb") as f:
                            img = Image.open(f)
                            if filter_shorter_edge and _shorter_edge_too_small(img, image_size):
                                filtered += 1
                                continue
                            img = _resize_shorter_edge_and_center_crop(img, image_size)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                    except Exception:
                        decode_errors += 1
                        continue
                    example = ExampleMsg(
                        features=FeaturesMsg(
                            feature={
                                "image": FeatureMsg(bytes_list=BytesListMsg(value=[img_bytes])),
                                "image/shape": FeatureMsg(int64_list=Int64ListMsg(value=[image_size, image_size, 3])),
                                "caption": FeatureMsg(
                                    bytes_list=BytesListMsg(
                                        value=[caption.encode("utf-8") for caption in caption_list]
                                    )
                                ),
                                "image_name": FeatureMsg(bytes_list=BytesListMsg(value=[str(image_name).encode("utf-8")])),
                            }
                        )
                    )
                    write(example.SerializeToString())
                    written += 1
            elif kind == "hf":
                spec, start, end = task[1], task[2], task[3]
                captions = CAPTIONS
                ds = hf_cache.get(spec)
                if ds is None:
                    ds = _load_hf_dataset(spec, num_proc=1)
                    ds = _disable_hf_image_decoding(ds)
                    hf_cache[spec] = ds
                subset = ds.select(range(start, end))
                for idx in range(len(subset)):
                    try:
                        entry = subset[idx]
                    except Exception:
                        decode_errors += 1
                        continue
                    image_name = _extract_hf_name(entry)
                    caption_list = _get_caption_list(image_name, captions)
                    if caption_list is None:
                        missing += 1
                        continue
                    image_source = _extract_hf_image_source(entry)
                    try:
                        img = _open_image_from_source(image_source)
                        if filter_shorter_edge and _shorter_edge_too_small(img, image_size):
                            filtered += 1
                            continue
                        img = _resize_shorter_edge_and_center_crop(img, image_size)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                    except Exception:
                        decode_errors += 1
                        continue
                    example = ExampleMsg(
                        features=FeaturesMsg(
                            feature={
                                "image": FeatureMsg(bytes_list=BytesListMsg(value=[img_bytes])),
                                "image/shape": FeatureMsg(int64_list=Int64ListMsg(value=[image_size, image_size, 3])),
                                "caption": FeatureMsg(
                                    bytes_list=BytesListMsg(
                                        value=[caption.encode("utf-8") for caption in caption_list]
                                    )
                                ),
                                "image_name": FeatureMsg(bytes_list=BytesListMsg(value=[str(image_name).encode("utf-8")])),
                            }
                        )
                    )
                    write(example.SerializeToString())
                    written += 1
            elif kind == "hf_indices":
                spec, indices = task[1], task[2]
                captions = CAPTIONS
                ds = hf_cache.get(spec)
                if ds is None:
                    ds = _load_hf_dataset(spec, num_proc=1)
                    ds = _disable_hf_image_decoding(ds)
                    hf_cache[spec] = ds
                subset = ds.select(indices)
                for idx in range(len(subset)):
                    try:
                        entry = subset[idx]
                    except Exception:
                        decode_errors += 1
                        continue
                    image_name = _extract_hf_name(entry)
                    caption_list = _get_caption_list(image_name, captions)
                    if caption_list is None:
                        missing += 1
                        continue
                    image_source = _extract_hf_image_source(entry)
                    try:
                        img = _open_image_from_source(image_source)
                        if filter_shorter_edge and _shorter_edge_too_small(img, image_size):
                            filtered += 1
                            continue
                        img = _resize_shorter_edge_and_center_crop(img, image_size)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                    except Exception:
                        decode_errors += 1
                        continue
                    example = ExampleMsg(
                        features=FeaturesMsg(
                            feature={
                                "image": FeatureMsg(bytes_list=BytesListMsg(value=[img_bytes])),
                                "image/shape": FeatureMsg(int64_list=Int64ListMsg(value=[image_size, image_size, 3])),
                                "caption": FeatureMsg(
                                    bytes_list=BytesListMsg(
                                        value=[caption.encode("utf-8") for caption in caption_list]
                                    )
                                ),
                                "image_name": FeatureMsg(bytes_list=BytesListMsg(value=[str(image_name).encode("utf-8")])),
                            }
                        )
                    )
                    write(example.SerializeToString())
                    written += 1
            else:
                raise ValueError(f"Unsupported task kind: {kind}")
    finally:
        writer.close()

    print(
        f"Shard {shard_idx + 1}/{total_shards} wrote {written} examples "
        f"(missing captions={missing}, decode errors={decode_errors}, filtered={filtered}).",
        flush=True,
    )
    return written, missing, decode_errors, filtered


def _write_metadata(output_dir: Path, image_size: int) -> None:
    import tensorflow_datasets as tfds
    from tensorflow_datasets.core.folder_dataset import compute_split_info_from_directory, write_metadata
    image_feature = tfds.features.Image(
        shape=(image_size, image_size, 3),
        encoding_format="png",
        doc="PNG-encoded RGB bytes (decode to uint8).",
    )
    features = tfds.features.FeaturesDict({
        "image": image_feature,
        "image/shape": tfds.features.Tensor(
            shape=(3,),
            dtype=tf.int64,
            doc="Height, width, channels.",
        ),
        "caption": tfds.features.Sequence(
            tfds.features.Text(
                doc="Captions for the image",
            ),
        ),
        "image_name": tfds.features.Text(
            doc="Image identifier (from filenames or dataset ids)",
        ),
    })
    split_infos = compute_split_info_from_directory(
        data_dir=str(output_dir)
    )
    write_metadata(
        data_dir=str(output_dir),
        features=features,
        split_infos=split_infos,
        version="1.0.0",
        description="dataset of image-caption pairs",
        supervised_keys=("image", "caption"),
    )
    print("Metadata written.", flush=True)


def create_tfrecord_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metadata_only:
        _write_metadata(output_dir, args.image_size)
        return
    if args.target_num_points is not None and args.target_num_points < 0:
        raise ValueError("--target_num_points must be >= 0")

    image_path = args.image_path
    captions_list = _load_hf_caption_maps(
        args.caption_subset,
        args.caption_columns,
        args.hf_num_proc,
    )

    shape_metadata = None
    if args.shape_metadata_path is not None:
        shape_metadata = _load_shape_metadata(args.shape_metadata_path)

    shard_tasks: List[List[Tuple]] = [[] for _ in range(args.num_shards)]
    folder_paths: Optional[List[Path]] = None
    folder_root: Optional[str] = None

    if Path(image_path).exists():
        root = Path(image_path)
        folder_paths = _iter_image_files(root)
        if not folder_paths:
            raise ValueError(f"No images found under {root}")
        total_points = len(folder_paths)
        if args.filter_shorter_edge and shape_metadata is not None:
            print("Using shape metadata to apply shorter-edge filtering for folder images.", flush=True)
            folder_paths, missing_shape_metadata = _filter_folder_paths_by_shape_metadata(
                folder_paths,
                str(root),
                shape_metadata,
                args.image_size,
            )
            total_points = len(folder_paths)
            if missing_shape_metadata:
                print(
                    f"Skipped {missing_shape_metadata} folder images missing shape metadata.",
                    flush=True,
                )
            print(
                f"Shape metadata kept {total_points} folder images with shorter edge >= {args.image_size}.",
                flush=True,
            )
            if args.target_num_points is not None and args.target_num_points < total_points:
                folder_paths = random.sample(folder_paths, args.target_num_points)
                print(
                    f"Randomly subsampled {len(folder_paths)} of {total_points} available examples.",
                    flush=True,
                )
        elif args.target_num_points is not None:
            if args.filter_shorter_edge:
                print(
                    f"Scanning {total_points} folder images to apply shorter-edge filtering before subsampling.",
                    flush=True,
                )
                folder_paths, total_points = _sample_folder_paths(
                    folder_paths,
                    args.target_num_points,
                    args.image_size,
                )
                if len(folder_paths) < total_points:
                    print(
                        f"Randomly subsampled {len(folder_paths)} of {total_points} available examples.",
                        flush=True,
                    )
                else:
                    print(
                        f"Shorter-edge filtering kept {total_points} folder images.",
                        flush=True,
                    )
            elif args.target_num_points < total_points:
                folder_paths = random.sample(folder_paths, args.target_num_points)
                print(
                    f"Randomly subsampled {len(folder_paths)} of {total_points} available examples.",
                    flush=True,
                )
        folder_root = str(root)
        splits = _split_indices(len(folder_paths), args.num_shards)
        for shard_idx, (start, end) in enumerate(splits):
            if start < end:
                shard_tasks[shard_idx].append(("folder", start, end))
    else:
        spec = image_path
        ds = _load_hf_dataset(spec, num_proc=args.hf_num_proc)
        total = len(ds)
        if args.filter_shorter_edge and shape_metadata is not None:
            print("Using shape metadata to apply shorter-edge filtering for dataset images.", flush=True)
            ds = _disable_hf_image_decoding(ds)
            eligible_indices, missing_shape_metadata = _filter_hf_indices_by_shape_metadata(
                ds,
                shape_metadata,
                args.image_size,
            )
            filtered_total = len(eligible_indices)
            if missing_shape_metadata:
                print(
                    f"Skipped {missing_shape_metadata} dataset images missing shape metadata.",
                    flush=True,
                )
            print(
                f"Shape metadata kept {filtered_total} dataset images with shorter edge >= {args.image_size}.",
                flush=True,
            )
            if args.target_num_points is not None and args.target_num_points < filtered_total:
                eligible_indices = random.sample(eligible_indices, args.target_num_points)
                print(
                    f"Randomly subsampled {len(eligible_indices)} of {filtered_total} available examples.",
                    flush=True,
                )
            splits = _split_indices(len(eligible_indices), args.num_shards)
            for shard_idx, (start, end) in enumerate(splits):
                if start < end:
                    shard_tasks[shard_idx].append(("hf_indices", spec, eligible_indices[start:end]))
        elif args.target_num_points is not None:
            if args.filter_shorter_edge:
                print(
                    f"Scanning {total} dataset images to apply shorter-edge filtering before subsampling.",
                    flush=True,
                )
                ds = _disable_hf_image_decoding(ds)
                sampled_indices, filtered_total = _sample_hf_indices(
                    ds,
                    args.target_num_points,
                    args.image_size,
                )
                if len(sampled_indices) < filtered_total:
                    print(
                        f"Randomly subsampled {len(sampled_indices)} of {filtered_total} available examples.",
                        flush=True,
                    )
                else:
                    print(
                        f"Shorter-edge filtering kept {filtered_total} dataset images.",
                        flush=True,
                    )
                splits = _split_indices(len(sampled_indices), args.num_shards)
                for shard_idx, (start, end) in enumerate(splits):
                    if start < end:
                        shard_tasks[shard_idx].append(("hf_indices", spec, sampled_indices[start:end]))
            elif args.target_num_points < total:
                sampled_indices = random.sample(range(total), args.target_num_points)
                print(
                    f"Randomly subsampled {len(sampled_indices)} of {total} available examples.",
                    flush=True,
                )
                splits = _split_indices(len(sampled_indices), args.num_shards)
                for shard_idx, (start, end) in enumerate(splits):
                    if start < end:
                        shard_tasks[shard_idx].append(("hf_indices", spec, sampled_indices[start:end]))
            else:
                splits = _split_indices(total, args.num_shards)
                for shard_idx, (start, end) in enumerate(splits):
                    if start < end:
                        shard_tasks[shard_idx].append(("hf", spec, start, end))
        else:
            splits = _split_indices(total, args.num_shards)
            for shard_idx, (start, end) in enumerate(splits):
                if start < end:
                    shard_tasks[shard_idx].append(("hf", spec, start, end))
        del ds

    shard_start = args.shard_start
    shard_end = args.shard_end if args.shard_end is not None else args.num_shards
    if shard_start < 0 or shard_end < 0 or shard_start > shard_end:
        raise ValueError(f"Invalid shard range: [{shard_start}, {shard_end})")
    if shard_end > args.num_shards:
        raise ValueError(f"--shard_end ({shard_end}) exceeds --num_shards ({args.num_shards})")

    global CAPTIONS, FOLDER_PATHS, FOLDER_ROOT
    CAPTIONS = captions_list
    FOLDER_PATHS = folder_paths
    FOLDER_ROOT = folder_root

    mp_ctx = (
        multiprocessing.get_context("fork")
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing.get_context("spawn")
    )
    use_fork = mp_ctx.get_start_method() == "fork"

    print(f"Writing TFRecords with {args.max_workers} parallel workers", flush=True)
    total_written = 0
    total_missing = 0
    total_decode_errors = 0
    total_filtered = 0
    try:
        if use_fork:
            executor = ProcessPoolExecutor(max_workers=args.max_workers, mp_context=mp_ctx)
        else:
            executor = ProcessPoolExecutor(
                max_workers=args.max_workers,
                mp_context=mp_ctx,
                initializer=_init_worker,
                initargs=(captions_list, folder_paths, folder_root),
            )
        with executor:
            futures = [
                executor.submit(
                    _write_tfrecord_shard,
                    shard_idx,
                    shard_tasks[shard_idx],
                    output_dir,
                    args.num_shards,
                    args.image_size,
                    args.filter_shorter_edge,
                )
                for shard_idx in range(shard_start, shard_end)
            ]
            for fut in as_completed(futures):
                written, missing, decode_errors, filtered = fut.result()
                total_written += int(written)
                total_missing += int(missing)
                total_decode_errors += int(decode_errors)
                total_filtered += int(filtered)
    finally:
        gc.collect()

    print(
        f"Finished writing {total_written} examples across {args.num_shards} TFRecords.",
        flush=True,
    )
    if total_missing:
        print(f"Skipped {total_missing} examples with missing captions.", flush=True)
    if total_decode_errors:
        print(f"Skipped {total_decode_errors} examples with decode errors.", flush=True)
    if total_filtered:
        print(f"Skipped {total_filtered} examples with shorter edge < image_size.", flush=True)

    if not args.skip_metadata:
        _write_metadata(output_dir, args.image_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--caption_subset", type=str, required=True, choices=[
        "fluxreason", "gptedit", "imagenet22k", "inaturalist", "megalith10m", "midjourneyv6", "pexels", "places365-challenge2016", "redcaps", "rendered_text", "textatlas", "yfcc"
    ])
    parser.add_argument("--caption_columns", type=str, nargs="+", default=["caption1", "caption2", "caption3", "caption4", "caption5"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--filter_shorter_edge", action="store_true")
    parser.add_argument("--num_shards", type=int, default=256)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--hf_num_proc", type=int, default=4)
    parser.add_argument("--shard_start", type=int, default=0)
    parser.add_argument("--shard_end", type=int, default=None)
    parser.add_argument("--skip_metadata", action="store_true")
    parser.add_argument("--metadata_only", action="store_true")
    parser.add_argument("--target_num_points", type=int, default=None)
    parser.add_argument("--shape_metadata_path", type=str, default=None)
    args = parser.parse_args()
    create_tfrecord_dataset(args)

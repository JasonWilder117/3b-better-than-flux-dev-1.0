import argparse
import os
import json
import glob
import io
import logging
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets, Image as HFImage
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

from vllm import LLM, SamplingParams
from tqdm import tqdm

def load_existing_stems(save_root):
    existing_files = set()
    consolidated_path = f"{save_root.rstrip('/')}.json"
    if os.path.isfile(consolidated_path):
        try:
            with open(consolidated_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    existing_files.update(data.keys())
                elif isinstance(data, list):
                    existing_files.update(data)
        except json.JSONDecodeError as err:
            logger.warning("Failed to parse %s: %s", consolidated_path, err)
    if os.path.isdir(save_root):
        for file in os.listdir(save_root):
            if not file.endswith(".json"):
                continue
            path = os.path.join(save_root, file)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        existing_files.update(data.keys())
                    elif isinstance(data, list):
                        existing_files.update(data)
            except json.JSONDecodeError as err:
                logger.warning("Failed to parse %s: %s", path, err)
    return existing_files

def filter_hf_dataset(hf_ds, existing_stems, num_proc=4):
    if not existing_stems:
        return hf_ds
    logger.info("Filtering Hugging Face dataset to exclude %d previously captioned items.", len(existing_stems))
    def _should_keep(example):
        stem = get_hf_stem(example)
        return stem not in existing_stems
    return hf_ds.filter(_should_keep, num_proc=num_proc)

def get_hf_stem(entry):
    if "global_id" in entry:
        return entry["global_id"]
    if "image_id" in entry:
        return entry["image_id"]
    if "__key__" in entry:
        return entry["__key__"]
    if "id" in entry:
        return entry["id"]
    if "image_path" in entry:
        return entry["image_path"]
    return None

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def to_pil_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return pil_loader(image["path"])
    if isinstance(image, str) and os.path.isfile(image):
        return pil_loader(image)
    raise TypeError(f"Unsupported image type: {type(image)}")

resample = Image.Resampling.BICUBIC
def resize_shorter_edge_and_center_crop(img: Image.Image, shorter_edge: int = 512) -> Image.Image:
    w, h = img.size
    short = min(w, h)
    if short > shorter_edge:
        if w <= h:
            new_w = shorter_edge
            new_h = int(round(h * (shorter_edge / w)))
        else:
            new_h = shorter_edge
            new_w = int(round(w * (shorter_edge / h)))
        img = img.resize((new_w, new_h), resample=resample)
        w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

class ImageDataset(Dataset):
    def __init__(self, args, existing_files=None):
        self.image_root = args.image_root
        extensions = ["*.jpg", "*.png", "*.JPEG", "*.webp", "*.jpeg"]
        all_image_paths = []
        for ext in extensions:
            all_image_paths.extend(glob.glob(os.path.join(args.image_root, "**", ext), recursive=True))
        self.image_paths = sorted(all_image_paths)[args.start_idx:args.end_idx]
        self.existing_files = existing_files if existing_files is not None else load_existing_stems(args.save_root)
        self.image_paths = [p for p in self.image_paths if self._get_stem(p) not in self.existing_files]

    def __len__(self):
        return len(self.image_paths)

    def _get_stem(self, image_path):
        if "places365-challenge2016" in self.image_root:
            return image_path.split("/")[-2] + "_" + image_path.split("/")[-1].split(".")[0]
        else:
            return image_path.split("/")[-1].split(".")[0]

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        stem:str = self._get_stem(image_path)
        try:
            img = pil_loader(image_path)
            img = resize_shorter_edge_and_center_crop(img, shorter_edge=512)
        except Exception as err:
            logger.warning("Skipping image %s due to load error: %s", image_path, err)
            return None
        return img, stem, "Describe the image in detail using one paragraph."

class HuggingFaceImageDataset(Dataset):
    def __init__(self, hf_dataset, dataset_type="default"):
        self.ds = hf_dataset
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        entry = self.ds[idx]
        stem = get_hf_stem(entry)
        if stem is None:
            raise NotImplementedError
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
        try:
            image_source = to_pil_image(image_source)
            image_source = resize_shorter_edge_and_center_crop(image_source, shorter_edge=512)
        except Image.DecompressionBombError as err:
            logger.warning("Skipping image %s due to decompression bomb check: %s", stem, err)
            return None
        except (UnidentifiedImageError, OSError, ValueError, TypeError) as err:
            logger.warning("Skipping image %s due to image decode error: %s", stem, err)
            return None
        return image_source, stem, self._build_prompt(entry)

    def _build_prompt(self, entry):
        if self.dataset_type == "rendered_text":
            text_lines = entry["json"]["ocr_annotation"]["text"]
            text_lines = [f"\"{item}\"" for item in text_lines]
            return (
                f"Describe the image in detail in one paragraph. For reference, there are {len(text_lines)} "
                f"lines of text in the image, and each line (comma-separated) is: {', '.join(text_lines)}. "
                "In your description, include the transcription, font, size, color, location, and rotation "
                "angle for the text."
            )
        if self.dataset_type == "textatlas":
            annotation = entry["annotation"]
            return f"Describe the image in detail in one paragraph. For reference, this is the ground truth annotation: \"{annotation}\""
        return "Describe the image in detail using one paragraph."

def collate_fn(batch):
    filtered = [item for item in batch if item is not None]
    skipped = len(batch) - len(filtered)
    if skipped:
        logger.debug("Collate function skipped %d problematic samples in the current batch.", skipped)
    if not filtered:
        return None, None, None
    imgs, stems, prompts = zip(*filtered)
    return list(imgs), list(stems), list(prompts)

def prepare_hf_dataset(hf_ds, args, existing_stems):
    start_idx = args.start_idx
    end_idx = min(args.end_idx, len(hf_ds)) if args.end_idx is not None else len(hf_ds)
    hf_ds = hf_ds.select(range(start_idx, end_idx))
    hf_ds = filter_hf_dataset(hf_ds, existing_stems)
    print(f"Processing {start_idx} to {end_idx} images")
    return hf_ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str)
    parser.add_argument("--save_root", type=str)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    existing_stems = load_existing_stems(args.save_root)
    logger.info("Loaded %d existing caption entries from %s", len(existing_stems), args.save_root)

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    llm = LLM(
        model="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        quantization="fp8",
        tensor_parallel_size=1,
        mm_encoder_tp_mode="data",
        enable_prefix_caching=False,
        max_num_seqs=512,
        limit_mm_per_prompt={"video": 0},
    )
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens)

    if args.image_root == "imagenet22k":
        hf_ds = load_dataset(
            "timm/imagenet-22k-wds",
            split="train",
            keep_in_memory=False,
            features=None,
            num_proc=4,
        )
        hf_ds = hf_ds.cast_column("jpg", HFImage(decode=False))
        hf_ds = prepare_hf_dataset(hf_ds, args, existing_stems)
        ds = HuggingFaceImageDataset(hf_ds)
    elif args.image_root == "pexels":
        hf_ds = load_dataset(
            "animetimm/pexels-tagger-v0-w640-ws-full",
            split="train",
            keep_in_memory=False,
            features=None,
            num_proc=4,
        )
        hf_ds = hf_ds.cast_column("webp", HFImage(decode=False))
        hf_ds = prepare_hf_dataset(hf_ds, args, existing_stems)
        ds = HuggingFaceImageDataset(hf_ds)
    elif args.image_root == "gptedit":
        hf_ds = load_dataset(
            "UCSC-VLAA/gpt-edit-simpler",
            split="train",
            keep_in_memory=False,
            features=None,
            num_proc=4,
        )
        hf_ds = hf_ds.cast_column("output", HFImage(decode=False))
        hf_ds = prepare_hf_dataset(hf_ds, args, existing_stems)
        ds = HuggingFaceImageDataset(hf_ds)
    elif args.image_root == "fluxreason":
        hf_ds = load_dataset(
            "LucasFang/FLUX-Reason-6M",
            split="train",
            keep_in_memory=False,
            features=None,
            num_proc=4,
        )
        hf_ds = hf_ds.cast_column("image", HFImage(decode=False))
        hf_ds = prepare_hf_dataset(hf_ds, args, existing_stems)
        ds = HuggingFaceImageDataset(hf_ds)
    elif args.image_root == "rendered_text":
        hf_ds = load_dataset(
            "wendlerc/RenderedText",
            split="train",
            keep_in_memory=False,
            features=None,
            num_proc=4,
        )
        hf_ds = hf_ds.cast_column("png", HFImage(decode=False))
        hf_ds = prepare_hf_dataset(hf_ds, args, existing_stems)
        ds = HuggingFaceImageDataset(hf_ds, dataset_type="rendered_text")
    elif args.image_root == "textatlas":
        parts = []
        for subset in ["CleanTextSynth", "PPT2Details", "PPT2Structured", "LongWordsSubset-A", "LongWordsSubset-M", "CoverBook", "Paper2Text", "TextVisionBlend", "StyledTextSynth", "TextScenesHQ"]:
            hf_ds = load_dataset(
                "CSU-JPG/TextAtlas5M",
                subset,
                split="train",
                keep_in_memory=False,
                features=None,
                num_proc=4,
            )
            hf_ds = hf_ds.cast_column("image", HFImage(decode=False))
            def add_global_id(example, subset_name=subset):
                example["global_id"] = f"{subset_name}:{example['image_path']}"
                return example
            hf_ds = hf_ds.map(add_global_id, num_proc=16)
            parts.append(hf_ds)
        hf_ds = concatenate_datasets(parts)
        hf_ds = prepare_hf_dataset(hf_ds, args, existing_stems)
        ds = HuggingFaceImageDataset(hf_ds, dataset_type="textatlas")
    else:
        print(f"Processing {args.start_idx} to {args.end_idx} images")
        ds = ImageDataset(args, existing_files=existing_stems)

    data_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False,
        collate_fn=collate_fn,
    )
    buffer_captions = {}
    for batch in tqdm(data_loader):
        imgs, stems, prompts = batch
        if imgs is None or stems is None or not len(stems) or prompts is None or not len(prompts):
            logger.debug("Skipping empty batch produced after filtering invalid samples.")
            continue
        batched_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_pil", "image_pil": img},
                    ],
                }
            ]
            for img, prompt in zip(imgs, prompts)
        ]
        try:
            responses = llm.chat(batched_messages, sampling_params=sampling_params)
        except Exception as e:
            logger.error("vLLM chat failed: %s", e)
            continue
        for stem, response in zip(stems, responses):
            caption = response.outputs[0].text.strip()
            buffer_captions[stem] = caption
            if len(buffer_captions) >= 500:
                with open(os.path.join(args.save_root, f'{list(buffer_captions.keys())[0]}.json'), 'w') as f:
                    f.write(json.dumps(buffer_captions))
                buffer_captions = {}
    if len(buffer_captions) > 0:
        with open(os.path.join(args.save_root, f'{list(buffer_captions.keys())[0]}.json'), 'w') as f:
            f.write(json.dumps(buffer_captions))
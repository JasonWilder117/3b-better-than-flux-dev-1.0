# Turn Image-Caption Pairs into TFRecords for Training

Our TPU data pipeline loads from datasets stored as TFRecords. [tfrecord.py](tfrecord.py) combines [our caption dataset](https://huggingface.co/datasets/zlab-princeton/i1-captions) with the image datasets to create TFRecords.

For each image dataset, run the following command:
```bash
# For Hugging Face image datasets (e.g., pexels):
python tfrecord.py --image_path pexels --caption_subset pexels --caption_columns caption1 caption2 caption3 caption4 caption5 --output_dir /path/to/save/tfrecords
# For images downloaded to a local folder (e.g., iNaturalist):
python tfrecord.py --image_path /path/to/inaturalist --caption_subset inaturalist --caption_columns caption1  --output_dir /path/to/save/tfrecords
```

Note that, while by default we load all five Qwen3-VL-30B-A3B captions (`caption1`, `caption2`, `caption3`, `caption4`, `caption5`), not all image datasets have all five sets of captions. Please refer to the [caption dataset viewer](https://huggingface.co/datasets/zlab-princeton/i1-captions) and adjust accordingly.

We store raw images with PNG compression, which can drastically reduce storage requirements compared to using raw images without PNG compression or using preprocessed VAE latents. Among these three options, we do not observe a significant difference in training speed on TPU machines.

Note that the number of TFRecords should be at least the number of parallel processes. This is because each parallel process is assigned an equal number of TFRecords by `tfds.even_splits`. For example, if you're using TPU v4-128 for training, which has 64 parallel processes, the number of TFRecords should be a multiple of 64.

For the same image dataset, we create separate sets of TFRecords for each resolution. At 512/1024 resolution, we discard images whose shorter edge is smaller than 512/1024 pixels, and subsample to 1_000_000 images (if not already below 1_000_000) using flags `--image_size=512/1024`, `--filter_shorter_edge` and `--target_num_points 1_000_000`.

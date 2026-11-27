# Data Processing Pipelines

This folder provides information and code for recreating the training dataset of i1.

## 1. Overview

Please follow [download_image](download_image) to download the images. Then, follow [make_tfrecord](make_tfrecord) to combine the images with [our caption dataset on Hugging Face](https://huggingface.co/datasets/zlab-princeton/i1-captions) into TFRecords. We provide the [synthetic_captioning](synthetic_captioning) code for completeness, but it is not needed for recreating i1's training dataset.

## 2. Folder Structure

This folder contains three independent subfolders. Their respective usage are listed below:

[download_image](download_image): instructions for downloading image datasets<br>
[make_tfrecord](make_tfrecord): instructions for creating TFRecords ready for training<br>
[synthetic_captioning](synthetic_captioning): our pipeline for captioning image datasets

## 3. TFRecord Datasets

The data processing pipelines result in a set of TFRecords for each dataset.

The processed TFRecords for the following datasets are hosted on [Hugging Face](https://huggingface.co/i1-datasets), you can directly download them for model training with the [PyTorch](../torch_train) or [JAX](../jax) code:

#### 256×256 Resolution Datasets
| dataset | link |
| --- | --- |
| FLUX-Reason |https://huggingface.co/datasets/zlab-princeton/i1-fluxreason-tfrecord|
| GPT-Edit |https://huggingface.co/datasets/zlab-princeton/i1-gptedit-tfrecord|
| ImageNet-22K |https://huggingface.co/datasets/zlab-princeton/i1-imagenet22k-tfrecord|
| iNaturalist |https://huggingface.co/datasets/zlab-princeton/i1-inaturalist-tfrecord|
| Midjourney v6 |https://huggingface.co/datasets/zlab-princeton/i1-midjourneyv6-tfrecord|
| Pexels |https://huggingface.co/datasets/zlab-princeton/i1-pexels-tfrecord|
| Places |https://huggingface.co/datasets/zlab-princeton/i1-places365-challenge2016-tfrecord|
| RenderedText |https://huggingface.co/datasets/i1-datasets/i1-rendered_text-tfrecord|
| TextAtlas |https://huggingface.co/datasets/zlab-princeton/i1-textatlas-tfrecord|

#### 512×512 Resolution Datasets
| dataset | link |
| --- | --- |
| FLUX-Reason |https://huggingface.co/datasets/i1-datasets/i1-fluxreason-512-resolution-1m-tfrecord|
| GPT-Edit |https://huggingface.co/datasets/i1-datasets/i1-gptedit-512-resolution-1m-tfrecord|
| ImageNet-22K |https://huggingface.co/datasets/i1-datasets/i1-imagenet22k-512-resolution-1m-tfrecord|
| Midjourney v6 |https://huggingface.co/datasets/i1-datasets/i1-midjourneyv6-512-resolution-1m-tfrecord|
| Pexels |https://huggingface.co/datasets/i1-datasets/i1-pexels-512-resolution-1m-tfrecord|
| Places |https://huggingface.co/datasets/i1-datasets/i1-places365-challenge2016-512-resolution-1m-tfrecord|
| RenderedText |https://huggingface.co/datasets/i1-datasets/i1-rendered_text-512-resolution-1m-tfrecord|
| TextAtlas |https://huggingface.co/datasets/i1-datasets/i1-textatlas-512-resolution-1m-tfrecord|

#### 1024×1024 Resolution Datasets
| dataset | link |
| --- | --- |
| FLUX-Reason |https://huggingface.co/datasets/i1-datasets/i1-fluxreason-1024-resolution-1m-tfrecord|
| GPT-Edit |https://huggingface.co/datasets/i1-datasets/i1-gptedit-1024-resolution-1m-tfrecord|
| Midjourney v6 |https://huggingface.co/datasets/i1-datasets/i1-midjourneyv6-1024-resolution-1m-tfrecord|
| TextAtlas |https://huggingface.co/datasets/i1-datasets/i1-textatlas-1024-resolution-1m-tfrecord|

We note that the processed TFRecord files for YFCC, RedCaps, and Megalith are currently not provided due to image license constraints. Please follow our data processing pipeline in this folder to download the images from URLs and combine them with [our captions](https://huggingface.co/datasets/zlab-princeton/i1-captions) into TFRecord files.
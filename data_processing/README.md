# Data Processing Pipelines

This folder provides information and code for recreating the training dataset of i1.

## 1. Overview

Please follow [download_image](download_image) to download the images. Then, follow [make_tfrecord](make_tfrecord) to combine the images with [our caption dataset on Hugging Face](https://huggingface.co/datasets/zlab-princeton/i1-captions) into TFRecords. We provide the [synthetic_captioning](synthetic_captioning) code for completeness, but it is not needed for recreating i1's training dataset.

## 2. Folder Structure

This folder contains three independent subfolders. Their respective usage are listed below:

[download_image](download_image): instructions for downloading image datasets<br>
[make_tfrecord](make_tfrecord): instructions for creating TFRecords ready for training<br>
[synthetic_captioning](synthetic_captioning): our pipeline for captioning image datasets

## 3. Reference TFRecord Dataset

The data processing pipelines result in a set of TFRecords for each dataset. The TFRecords corresponding to GPT-Edit is hosted on [Hugging Face](https://huggingface.co/datasets/zlab-princeton/i1-gptedit-tfrecord) as a point of reference.
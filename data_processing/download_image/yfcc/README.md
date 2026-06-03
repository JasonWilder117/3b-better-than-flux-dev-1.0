# YFCC100M Image Download

This directory contains instructions and code for downloading images from the YFCC100M dataset.

## 1. Download metadata

Download the official metadata file once. It contains URLs and `photoid` values for the images.

```bash
wget -O /path/to/save/yfcc100m_dataset.sql https://multimedia-commons.s3-us-west-2.amazonaws.com/tools/etc/yfcc100m_dataset.sql
```

## 2. Download images

Launch independent download scripts for different shards. Images are saved as `{photoid}.jpg`; rows without a `photoid` are skipped (`photoid` is a field in the image metadata).

```bash
# Example: four parallel runs
for SHARD in 0 1 2 3; do
  python download.py \
    --metadata_path /path/to/saved/yfcc100m_dataset.sql \
    --save_dir /folder/to/save/images \
    --workers 8 \
    --num_shards 4 \
    --shard_id ${SHARD} \
    > log_shard${SHARD}.txt 2>&1 &
done
```
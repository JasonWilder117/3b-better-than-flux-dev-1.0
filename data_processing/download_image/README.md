# Download Image Data

This document provides commands and instructions for downloading each dataset we use.

## 1. ImageNet-22K
This dataset can be accessed on Hugging Face.
```python
from datasets import load_dataset
ds = load_dataset("timm/imagenet-22k-wds", keep_in_memory=False)
```

## 2. YFCC
Please follow the instructions in the [yfcc](yfcc) folder.

## 3. RedCaps
Environment setup:
```bash
git clone https://github.com/redcaps-dataset/redcaps-downloader
cd redcaps-downloader
conda create -n redcaps python=3.9
conda activate redcaps
pip install -r requirements.txt
python setup.py develop
mkdir /path/to/store/redcaps
cd /path/to/store/redcaps
wget https://huggingface.co/datasets/kdexd/red_caps/resolve/main/data/redcaps_v1.0_annotations.zip
unzip redcaps_v1.0_annotations.zip
```

Download images:
```bash
for ann_file in /path/to/store/redcaps/annotations/*.json; do
    redcaps download-imgs -a $ann_file --save-to /path/to/store/redcaps/images --resize -1 -j 16
done
```


## 4. Megalith
This dataset can be accessed on Hugging Face.
```python
from datasets import load_dataset
ds = load_dataset("madebyollin/megalith-10m", keep_in_memory=False)
```

## 5. Places
Please follow the instructions at http://places2.csail.mit.edu/download.html to download the dataset.
```bash
wget http://data.csail.mit.edu/places/places365/train_large_places365challenge.tar
tar -xf train_large_places365challenge.tar
```

## 6. Pexels
This dataset can be accessed on Hugging Face.
```python
from datasets import load_dataset
ds = load_dataset("animetimm/pexels-tagger-v0-w640-ws-full", keep_in_memory=False)
```

## 7. iNaturalist
Please follow the instructions at https://github.com/inquire-benchmark/INQUIRE/tree/main/data to download the dataset.
```bash
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2024/train.tar.gz
tar -xzf train.tar.gz
```

## 8. FLUX-Reason
This dataset can be accessed on Hugging Face.
```python
from datasets import load_dataset
ds = load_dataset("LucasFang/FLUX-Reason-6M", keep_in_memory=False)
```

## 9. Midjourney v6
This dataset can be accessed on Hugging Face.
```python
from datasets import load_dataset
ds = load_dataset("Photoroom/midjourney-v6-recap", keep_in_memory=False)
```

## 10. GPT-Edit
This dataset can be accessed on Hugging Face.
```python
from datasets import load_dataset
ds = load_dataset("UCSC-VLAA/gpt-edit-simpler", keep_in_memory=False)
```

## 11. TextAtlas
This dataset can be accessed on Hugging Face.
```python
from datasets import load_dataset

for sub_dataset in ["CleanTextSynth", "PPT2Details", "PPT2Structured", "LongWordsSubset-A", "LongWordsSubset-M", "CoverBook", "Paper2Text", "TextVisionBlend", "StyledTextSynth", "TextScenesHQ"]:
    ds = load_dataset("CSU-JPG/TextAtlas5M", sub_dataset, keep_in_memory=False)
```

## 12. Rendered Text
This dataset can be accessed on Hugging Face.
```python
from datasets import load_dataset
ds = load_dataset("wendlerc/RenderedText", keep_in_memory=False)
```
# Synthetic Captioning of Image Datasets

## 1. Environment setup
```bash
conda create -n qwen3vl python=3.11 -y
conda activate qwen3vl
pip install --no-cache-dir "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" "torchaudio==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128
pip install --no-build-isolation "flash-attn==2.8.3"
pip install --no-cache-dir accelerate qwen-vl-utils==0.0.14
pip install vllm==0.11.0
conda install -c huggingface -c conda-forge datasets
pip install --no-cache-dir flashinfer-python
pip install --no-cache-dir flashinfer-cubin
pip install --no-cache-dir flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128
```

## 2. Captioning
```bash
# For Hugging Face image datasets (e.g., pexels):
python caption.py --image_root pexels --save_root /path/to/save/captions
# For images downloaded to a local folder (e.g., iNaturalist):
python caption.py --image_root /path/to/image/folder --save_root /path/to/save/captions
```
[caption.py](caption.py) creates dictionaries that map from an image identifier to the corresponding caption. To avoid overpopulating the file system, we group 500 captions into one `.json` dictionary when saving the captions. The identifier used for each dataset is listed below:

|dataset|type|identifier|
|:-|:-:|:-:|
|ImageNet-22K|Hugging Face|{\_\_key\_\_}|
|YFCC|Image Folder|{name of the image}|
|RedCaps|Image Folder|{name of the image}|
|Megalith|Hugging Face|{image_id}|
|Places|Image Folder|{name of the folder}_{name of the image}|
|Pexels|Hugging Face|{\_\_key\_\_}|
|iNaturalist|Image Folder|{name of the image}|
|FLUX-Reason|Hugging Face|{id}|
|Midjourney v6|Hugging Face|{id}_0 / {id}_1 / {id}_2 / {id}_3|
|GPT-Edit|Hugging Face|{id}|
|TextAtlas|Hugging Face|{huggingface_subset_name}:{the "image_path" column's value}|
|Rendered Text|Hugging Face|{\_\_key\_\_}|
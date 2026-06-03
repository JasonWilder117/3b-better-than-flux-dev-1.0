# JAX Training and Inference on TPUs

## 1. Prerequisite

### 1.1 Data
Please make sure that you've followed the [data processing guide](../data_processing) and uploaded the TFRecords to a Google Cloud bucket.
After uploading, please update the data path to each dataset in the config files under [configs](configs).

```python
path_and_count = [
    ('gs://path/to/dataset_1/tfrecord', weight_for_dataset_1),
    ('gs://path/to/dataset_2/tfrecord', weight_for_dataset_2),
    ('gs://path/to/dataset_3/tfrecord', weight_for_dataset_3),
]
```

### 1.2 T5Gemma Checkpoints

Download the T5Gemma weights from kaggle
```python
import kagglehub
import os
os.environ["KAGGLEHUB_CACHE"] = "."
path = kagglehub.model_download("google/t5gemma/flax/t5gemma-2b-2b-ul2-it")
print(path)
```

Upload the T5Gemma weights to a Google Cloud bucket
```
gsutil -m cp -r path_from_the_command_above gs://path/to/save/t5gemma/checkpoint
```

Update the T5Gemma checkpoint path in [text_encoder/text_encoder.py](text_encoder/text_encoder.py).
```python
t5gemma_ckpt = "gs://path/to/t5gemma/checkpoint"
```

### 1.3 TPU Access
Instructions below assume that you already have access to a TPU machine. You can apply for free TPU access through the [TPU Research Cloud program](https://sites.research.google/trc/about/) and learn how to use TPUs through these guides: https://docs.cloud.google.com/tpu/docs/intro-to-tpu, https://github.com/ayaka14732/tpu-starter, and https://github.com/boyazeng/tpu_intro.

Please set environment variables to match your TPU information:
```bash
export PROJECT_ID=your_project_id
export ZONE=your_gcp_zone
export TPU_NAME=your_tpu_name
```

## 2. Folder Structure
[configs](configs): training and inference config files (each file defines one model/training setup).<br>
[datasets](datasets): input pipeline code for TFRecords.<br>
[diffusion](diffusion): rectified-flow utilities and inference pipeline.<br>
[inference](inference): inference entry point and benchmark prompt sets.<br>
[models](models): DiT backbone implementations.<br>
[scripts](scripts): example shell scripts, including environment setup, training, and sampling.<br>
[text_encoder](text_encoder): modular implementation of text encoders.<br>
[training](training): training entry point.<br>
[utils](utils): checkpointing, metric writing, and data preprocessing utilities.<br>
[vae](vae): modular implementation of VAEs.<br>

## 3. Environment Setup
[setup.sh](scripts/setup.sh) (attached below) pulls this codebase, installs miniconda, and installs the conda environment for JAX/TPU training.
```
# Clone the repo
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --ssh-key-file=~/.ssh/google_compute_engine --worker=all \
--command "git clone https://github.com/zlab-princeton/i1"

# Install miniconda
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --ssh-key-file=~/.ssh/google_compute_engine --worker=all \
--command "mkdir -p ~/miniconda3 && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
rm ~/miniconda3/miniconda.sh && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda init"

# Install conda environment
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --ssh-key-file=~/.ssh/google_compute_engine --worker=all \
--command "cd i1/jax && bash scripts/prepare_env.sh"
```
To install a GPU-compatible environment, you can simply replace `jax[tpu]` in [prepare_env.sh](scripts/prepare_env.sh) with `jax[cuda]`.

Note that you may want to replace the `git clone https://github.com/zlab-princeton/i1` line with a clone of your own updated repo containing the updated data paths in [configs](configs/) and updated T5Gemma weight paths in [text_encoder/text_encoder.py](text_encoder/text_encoder.py).

## 4. Training
[train.sh](scripts/train.sh) contains an example training command (attached below) for training the cross-attention baseline.
```bash
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --ssh-key-file=~/.ssh/google_compute_engine --worker=all \
--command "cd i1/jax && \
export TF_CPP_MIN_LOG_LEVEL=2 && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate i1_jax_train && \
python -m pip install -U wandb && \
wandb login your_wandb_token && \
hf auth login --token your_huggingface_token && \
python3 -m training.main \
--config=configs/controlled_exp_baselines/cross_attn.py \
--auto_generate_on_ckpt=True \
--workdir=gs://path/to/save/checkpoints/and/images"
```
To train a different baseline, you only need to set `--config=configs/controlled_exp_baselines/single_stream.py` or `--config=configs/controlled_exp_baselines/dual_stream.py`.

To reproduce i1-3B, use `--config=configs/i1_training/256_resolution.py`, `--config=configs/i1_training/512_resolution.py`, and ``--config=configs/i1_training/1024_resolution.py`` for 256, 512, and 1024-resolution training, respectively. Note that for 512/1024-resolution training, you'd need to provide the trained 256/512-resolution checkpoint via the `--config.resume` flag.

By default, to avoid high cross-region data transfer cost, the code includes a checker to ensure the data, TPU, and workdir are in the same region. You can disable this checker using `--check_gcs_region=False`.

`--auto_generate_on_ckpt` controls whether to run generation for prompts `["dpg", "prism_simple_rewrite", "longtext"]` and save the zipped images to `--workdir` whenever a permanent checkpoint is created.

## 5. Inference
While the `--auto_generate_on_ckpt` flag of [main.py](training/main.py) supports generating images during the training process for fixed inference settings (*i.e.*, the setting in our controlled experiments: CFG=12, CFG rescale=0, and steps=250), [generate.py](inference/generate.py) allows running inference for an arbitrary checkpoint in an arbitrary setting.

Generate images and copy to your local machine:
```bash
cd scripts
# The process of copying the generated images out to a local machine requires zip
bash install_zip.sh
bash sample.sh
```

## 6. Miscellaneous
If you would like to interrupt your running process on TPU:
```
cd scripts
bash kill.sh
```
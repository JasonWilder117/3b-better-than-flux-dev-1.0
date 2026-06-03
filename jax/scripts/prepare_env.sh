export KERAS_BACKEND=tensorflow
unset JAX_PLATFORM_NAME
source ~/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n i1_jax_train python=3.11 -y
conda activate i1_jax_train
python -m pip install -U pip
python -m pip install "jax[tpu]==0.6.2" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -r requirements.txt
python -m pip install --no-deps tensorflow-addons==0.23.0 tfa-nightly==0.23.0.dev20240415222534
python -m pip install torch==2.10.0+cpu torchvision==0.25.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
sed -i 's/^from huggingface_hub import cached_download, hf_hub_download, model_info$/from huggingface_hub import hf_hub_download, model_info/' ~/miniconda3/envs/i1_jax_train/lib/python3.11/site-packages/diffusers/utils/dynamic_modules_utils.py
sed -i 's/metadata = dict(metadata)/if dataclasses.is_dataclass(metadata) and not isinstance(metadata, dict):\n    metadata = dataclasses.asdict(metadata)\n    return metadata['"'"'item_metadata'"'"'], path\n  else:\n    metadata = dict(metadata)/' $(python -c "import gemma.gm.ckpts._checkpoint as m; print(m.__file__)")
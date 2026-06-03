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
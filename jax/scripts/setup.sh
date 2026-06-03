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
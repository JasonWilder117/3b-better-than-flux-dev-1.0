gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --ssh-key-file=~/.ssh/google_compute_engine \
  --worker=all \
  --project=$PROJECT_ID \
  --command '
    sudo systemctl stop unattended-upgrades || true

    # Make sure no lingering dpkg/apt processes are running
    if pgrep -x unattended-upgr >/dev/null; then
      echo "unattended-upgrades still running; killing..."
      sudo pkill -x unattended-upgr || true
    fi

    # Only remove lock after processes are gone
    if sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
      echo "Lock still in use; aborting to avoid corruption."
      exit 1
    fi

    sudo apt-get update &&
    sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y zip
  '

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --ssh-key-file=~/.ssh/google_compute_engine --worker=all \
--project=$PROJECT_ID \
--command "which zip"
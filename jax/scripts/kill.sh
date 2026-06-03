gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --worker=all \
    --command='
  # Kill all python processes, but keep checking until we observe a stable
  # "no python processes" window (processes can respawn after the first kill).
  host="$(hostname)"
  log() { echo "[$host] $*"; }

  stable_required_seconds=30
  sleep_interval_seconds=1
  max_rounds=60

  log "Finding Python processes..."
  stable_seconds=0
  round=0
  while (( round < max_rounds )); do
    round=$((round + 1))

    pids="$(pgrep python 2>/dev/null || true)"
    if [[ -z "$pids" ]]; then
      stable_seconds=$((stable_seconds + sleep_interval_seconds))
      log "No Python processes found (stable ${stable_seconds}/${stable_required_seconds}s)"
      if (( stable_seconds >= stable_required_seconds )); then
        log "All Python processes killed"
        break
      fi
    else
      stable_seconds=0
      log "Found Python processes: $pids"
      for pid in $pids; do
        log "Killing process ${pid}..."
        sudo kill -9 "${pid}" 2>/dev/null || true
      done
    fi

    sleep "${sleep_interval_seconds}"
  done

  if (( stable_seconds < stable_required_seconds )); then
    remaining_pids="$(pgrep python 2>/dev/null || true)"
    if [[ -n "$remaining_pids" ]]; then
      log "WARNING: Python processes still present after ${max_rounds} rounds: ${remaining_pids}"
    else
      log "WARNING: Python processes kept respawning; could not observe ${stable_required_seconds}s clean window within ${max_rounds} rounds"
    fi
  fi

  # Also check for TPU device locks
  for device_name in vfio/ accel; do
    pids="$(sudo lsof -t /dev/${device_name}* 2>/dev/null | sort -u || true)"
    if [[ -n "${pids}" ]]; then
      log "Found processes on /dev/${device_name}*: ${pids}"
      for pid in ${pids}; do
        sudo kill -9 "${pid}" 2>/dev/null || true
      done
    fi
  done

  # Cleanup lockfile
  sudo rm -f /tmp/libtpu_lockfile
  log "TPU cleanup complete"
  '
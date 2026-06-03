import importlib.util
import os
import shutil
from types import SimpleNamespace
import zipfile

from absl import logging
import jax
from jax.experimental import multihost_utils
import numpy as np
from tensorflow.io import gfile

def _zip_directory(source_dir, output_zip_path):
    if os.path.exists(output_zip_path):
        os.remove(output_zip_path)
    source_parent = os.path.dirname(source_dir)
    with zipfile.ZipFile(output_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                arcname = os.path.relpath(full_path, source_parent)
                zipf.write(full_path, arcname)


class InProcessAutoGenerator:
    def __init__(
        self,
        enabled,
        train_config,
        save_ckpt_path,
        per_proc_batch_size=256,
    ):
        self.enabled = bool(enabled)
        if not self.enabled:
            return
        self.train_config = train_config
        self.save_ckpt_path = save_ckpt_path
        self._generate_module = None
        self.per_proc_batch_size = int(per_proc_batch_size)

        self.prompt_types = ["dpg", "prism_simple_rewrite", "longtext"]
        self.cfg_scales = [12.0]
        self.cfg_rescales = [0.0]
        self.num_sampling_steps = 250

    def _expected_zip_output_paths(self, step):
        padded_step = f"{int(step):09d}"
        ckpt_parent = os.path.dirname(self.save_ckpt_path.rstrip("/"))
        for prompt_type in self.prompt_types:
            for cfg_scale in self.cfg_scales:
                for cfg_rescale in self.cfg_rescales:
                    dest_dir = os.path.join(
                        ckpt_parent,
                        prompt_type,
                        f"{float(cfg_scale):.1f}",
                        f"{float(cfg_rescale):.1f}",
                        f"{prompt_type}_samples_{padded_step}",
                    )
                    for worker_idx in range(jax.process_count()):
                        yield os.path.join(dest_dir, f"worker{worker_idx}.zip")

    def is_step_completed(self, step):
        if not self.enabled:
            return False
        return all(gfile.exists(path) for path in self._expected_zip_output_paths(step))

    def _load_generate_module(self):
        if self._generate_module is not None:
            return self._generate_module
        module_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "inference", "generate.py")
        )
        spec = importlib.util.spec_from_file_location("inference_generate", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load inference generate module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._generate_module = module
        return module

    def _wait_for_checkpoint_ready(self, step, ckpt_writer):
        ckpt_ready = np.array([1], dtype=np.int32)
        if jax.process_index() == 0:
            try:
                if ckpt_writer is not None:
                    ckpt_writer.get()
                ckpt_step_path = f"{self.save_ckpt_path}-{int(step):09d}"
                if not gfile.exists(ckpt_step_path):
                    raise FileNotFoundError(f"Expected step checkpoint does not exist: {ckpt_step_path}")
            except Exception as e:
                logging.exception("Checkpoint readiness failed at step %d: %s", step, e)
                ckpt_ready[0] = 0

        ckpt_ready_all = multihost_utils.process_allgather(ckpt_ready)
        if int(np.min(np.asarray(ckpt_ready_all))) == 0:
            raise RuntimeError(f"Checkpoint write/visibility failed for step {step}; aborting auto-generation.")

    def _run_generation_and_upload_for_process(self, step):
        generate_module = self._load_generate_module()
        padded_step = f"{int(step):09d}"
        for prompt_type in self.prompt_types:
            sample_dir_base = f"{prompt_type}_samples"
            gen_args = SimpleNamespace(
                sample_dir=sample_dir_base,
                per_proc_batch_size=self.per_proc_batch_size,
                image_size=self.train_config.image_size,
                cfg_scale=list(self.cfg_scales),
                cfg_rescale=list(self.cfg_rescales),
                checkpoint_iters=[int(step)],
                num_sampling_steps=self.num_sampling_steps,
                global_seed=42,
                ckpt=self.save_ckpt_path,
                prompt_type=prompt_type,
                sync_after_sampling=False,
            )
            generate_module.run_with_config(gen_args, self.train_config)

            for cfg_scale in self.cfg_scales:
                for cfg_rescale in self.cfg_rescales:
                    run_dir = (
                        f"{sample_dir_base}_{padded_step}_cfg{float(cfg_scale):.1f}"
                        f"_rescale{float(cfg_rescale):.1f}"
                    )
                    if not os.path.isdir(run_dir):
                        raise FileNotFoundError(
                            f"Expected generated sample directory does not exist: {run_dir}"
                        )
                    zip_path = f"{run_dir}.zip"
                    _zip_directory(run_dir, zip_path)
                    worker_suffix = f"worker{jax.process_index()}.zip"
                    ckpt_parent = os.path.dirname(self.save_ckpt_path.rstrip("/"))
                    dest_dir = os.path.join(
                        ckpt_parent,
                        prompt_type,
                        f"{float(cfg_scale):.1f}",
                        f"{float(cfg_rescale):.1f}",
                        f"{prompt_type}_samples_{padded_step}",
                    )
                    gfile.makedirs(dest_dir)
                    gfile.copy(zip_path, os.path.join(dest_dir, worker_suffix), overwrite=True)
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                    if os.path.isdir(run_dir):
                        shutil.rmtree(run_dir)

    def run_after_checkpoint(self, step, ckpt_writer):
        if not self.enabled:
            return
        self._wait_for_checkpoint_ready(step, ckpt_writer)
        auto_generate_failed = np.array([0], dtype=np.int32)
        try:
            self._run_generation_and_upload_for_process(step)
        except Exception as e:
            logging.exception("Auto-generation failed at step %d: %s", step, e)
            auto_generate_failed[0] = 1
        all_hosts_status = multihost_utils.process_allgather(auto_generate_failed)
        if int(np.max(np.asarray(all_hosts_status))) != 0:
            raise RuntimeError(f"Auto-generation failed for step {step}; aborting training.")

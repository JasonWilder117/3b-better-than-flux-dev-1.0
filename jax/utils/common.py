import collections
import contextlib
import dataclasses
import io
import json
import multiprocessing
import multiprocessing.pool
import os
import re
import time
from typing import Mapping

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np

import tensorflow.io.gfile as gfile


def npload(fname):
    if os.path.exists(fname):
        loaded = np.load(fname, allow_pickle=False)
    elif _is_gs_path(fname):
        from google.cloud import storage

        bucket_name, blob_name = _parse_gs_path(fname)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        loaded = np.load(
            io.BytesIO(blob.download_as_bytes(timeout=3600)),
            allow_pickle=False)
    else:
        with gfile.GFile(fname, "rb") as f:
            data = f.read()
        loaded = np.load(io.BytesIO(data), allow_pickle=False)

    if isinstance(loaded, np.ndarray):
        return loaded
    else:
        return dict(loaded)


def load_checkpoint(tree, npz):
    if isinstance(npz, str):
        npz = npload(npz)
    keys, values = zip(*list(npz.items()))
    if tree:
        checkpoint = tree.unflatten(values)
    else:
        checkpoint = recover_tree(keys, values)
    checkpoint = jax.tree_util.tree_map(recover_dtype, checkpoint)
    return checkpoint


def itstime(step, every_n_steps, total_steps, host=None, last=True, first=True,
            drop_close_to_last=0.25):

    close_to_last = False
    if drop_close_to_last and every_n_steps:
        close_to_last = abs(step - total_steps) < drop_close_to_last * every_n_steps

    is_host = host is None or jax.process_index() == host
    is_step = every_n_steps and (step % every_n_steps == 0) and not close_to_last
    is_last = every_n_steps and step == total_steps
    is_first = every_n_steps and step == 1
    return is_host and (is_step or (last and is_last) or (first and is_first))


def checkpointing_timeout(writer, timeout):
    if writer is not None:
        try:
            writer.get(timeout=timeout)
        except multiprocessing.TimeoutError as e:
            raise TimeoutError("Checkpoint writing timed out. If this is expected, increase `ckpt_timeout`.") from e


def hms(s):
    if s < 60:
        return f"{s:.0f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m:.0f}m{s:.0f}s"
    h, m = divmod(m, 60)
    return f"{h:.0f}h{m:.0f}m"


class Chrono:

    def __init__(self):
        self._timing_history = collections.defaultdict(list)
        self._measure = None
        self._write_note = None

        self.program_start_time = time.monotonic()
        self.train_start_time = None
        self.train_start_step = None

        self.prev_time = None
        self.prev_step = None

        self.pause_start = None
        self.paused_time = 0

        self.total_steps = None
        self.global_bs = None
        self.steps_per_epoch = None

        self.warmup = 2
        self.load()
        self.note = "Chrono n/a"

    def inform(self, *, first_step=None, total_steps=None, global_bs=None,
                steps_per_epoch=None, measure=None, write_note=None):
        self.prev_step = first_step if first_step is not None else self.prev_step
        self.total_steps = total_steps or self.total_steps
        self.steps_per_epoch = steps_per_epoch or self.steps_per_epoch
        self.global_bs = global_bs or self.global_bs
        self._measure = measure or self._measure
        self._write_note = write_note or self._write_note
        if self.total_steps and self.prev_step is not None:
            self.note = (f"Steps:{self.prev_step}/{self.total_steps} "
                            f"[{self.prev_step/self.total_steps:.1%}]")

    def tick(self, step, measure=None, write_note=None):
        if step == self.prev_step: return

        measure = measure or self._measure
        write_note = write_note or self._write_note

        now = time.monotonic()
        measure("uptime", now - self.program_start_time)
        self.flush_timings()

        ds = step - self.prev_step
        self.prev_step = step
        self.accum_examples_seen += ds * self.global_bs
        measure("examples_seen", self.accum_examples_seen)
        measure("progress", step / self.total_steps)
        if self.steps_per_epoch:
            measure("epoch", step / self.steps_per_epoch)

        if self.warmup > 1:
            self.warmup -= 1
            write_note(self.note)
            return
        if self.warmup == 1:
            self.train_start_time = self.prev_time = now
            self.train_start_step = step
            self.accum_program_time += now - self.program_start_time
            self.paused_time = 0
            self.warmup = 0
            write_note(self.note)
            return

        dt = now - self.prev_time - self.paused_time
        ncores = jax.device_count()
        measure("img/sec/core", self.global_bs * ds / dt / ncores)

        self.accum_train_time += dt
        self.accum_pause_time += self.paused_time
        self.accum_program_time += dt + self.paused_time

        core_hours = self.accum_train_time * ncores / 60 / 60
        devtype = jax.devices()[0].device_kind
        measure(f"core_hours_{devtype}", core_hours)
        measure("core_hours", core_hours)

        dt = now - self.train_start_time
        steps_timed = step - self.train_start_step
        steps_todo = self.total_steps - step
        self.note = f"Steps:{step}/{self.total_steps} [{step/self.total_steps:.1%}]"
        self.note += f"\nWalltime:{hms(self.accum_program_time)}"
        self.note += f" ({hms(self.accum_pause_time)} eval)"
        self.note += f"\nETA:{hms(dt / steps_timed * steps_todo)}"
        self.note += f"\nTotal train time:{hms(dt / steps_timed * self.total_steps)}"
        write_note(self.note)

        self.prev_time = now
        self.paused_time = 0

    def pause(self, wait_for=()):
        assert self.pause_start is None, "Don't pause twice."
        jax.block_until_ready(wait_for)
        self.pause_start = time.monotonic()

    def resume(self):
        self.paused_time += time.monotonic() - self.pause_start
        self.pause_start = None

    def save(self):
        return dict(
            accum_program_time=self.accum_program_time,
            accum_train_time=self.accum_train_time,
            accum_pause_time=self.accum_pause_time,
            accum_examples_seen=self.accum_examples_seen,
        )

    def load(self, ckpt=None):
        ckpt = ckpt or {}
        self.accum_program_time = ckpt.get("accum_program_time", 0.0)
        self.accum_train_time = ckpt.get("accum_train_time", 0.0)
        self.accum_pause_time = ckpt.get("accum_pause_time", 0.0)
        self.accum_examples_seen = ckpt.get("accum_examples_seen", 0)

    @contextlib.contextmanager
    def log_timing(self, name, *, noop=False):
        t0 = time.monotonic()
        yield
        dt = time.monotonic() - t0
        if not noop:
            self._measure(name, dt)
            logging.info("TIMING[%s]: %s", name, dt)
            logging.flush()

    def flush_timings(self):
        for name, times in self._timing_history.items():
            self._measure(name, np.mean(times))
        self._timing_history.clear()


chrono = Chrono()


def _traverse_with_names(tree, with_inner_nodes=False):
    if dataclasses.is_dataclass(tree):
        tree = flax.serialization.to_state_dict(tree)
    if tree is None:
        return
    elif isinstance(tree, Mapping):
        keys = sorted(tree.keys())
        for key in keys:
            for path, v in _traverse_with_names(tree[key], with_inner_nodes):
                yield (key + "/" + path).rstrip("/"), v
        if with_inner_nodes:
            yield "", tree
    elif isinstance(tree, (list, tuple)):
        for idx in range(len(tree)):
            for path, v in _traverse_with_names(tree[idx], with_inner_nodes):
                yield (str(idx) + "/" + path).rstrip("/"), v
        if with_inner_nodes:
            yield "", tree
    else:
        yield "", tree


def tree_flatten_with_names(tree):
    vals, tree_def = jax.tree_util.tree_flatten(tree)

    tokens = range(len(vals))
    token_tree = tree_def.unflatten(tokens)
    val_names, perm = zip(*_traverse_with_names(token_tree))
    inv_perm = np.argsort(perm)

    assert len(val_names) == len(vals)

    return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def tree_map_with_names(f, tree, *rest):
    names_and_vals, tree_def = tree_flatten_with_names(tree)
    names, vals = zip(*names_and_vals)
    rest_vals = [list(zip(*tree_flatten_with_names(t)[0]))[1] for t in rest]
    vals = [f(*name_and_vals) for name_and_vals in zip(names, vals, *rest_vals)]
    return tree_def.unflatten(vals)


def recover_dtype(a):
    if hasattr(a, "dtype") and a.dtype.type is np.void:
        assert a.itemsize == 2, "Unknown dtype!"
        return a.view(jax.numpy.bfloat16)
    else:
        return a


def _is_gs_path(path):
    return path.startswith("gs://")


def _parse_gs_path(path):
    path = path.removeprefix("gs://")
    bucket_name, _, blob_name = path.partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Expected a gs://bucket/path checkpoint path, got {path!r}")
    return bucket_name, blob_name


def _rewrite_gcs_blob(source_blob, dest_blob, timeout=3600):
    token = None
    while True:
        token, _, _ = dest_blob.rewrite(source_blob, token=token, timeout=timeout)
        if token is None:
            return


def save_checkpoint(checkpoint, path, step_copy=None, compressed=False):
    names_and_vals, _ = tree_flatten_with_names(checkpoint)
    io_buffer = io.BytesIO()

    if compressed:
        np.savez_compressed(io_buffer, **{k: v for k, v in names_and_vals})
    else:
        np.savez(io_buffer, **{k: v for k, v in names_and_vals})

    if _is_gs_path(path):
        from google.cloud import storage

        bucket_name, blob_name = _parse_gs_path(path)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        temp_blob = bucket.blob(blob_name + "-TEMPORARY")
        temp_blob.chunk_size = 256 * 1024 * 1024
        io_buffer.seek(0)
        temp_blob.upload_from_file(
            io_buffer,
            size=io_buffer.getbuffer().nbytes,
            content_type="application/octet-stream",
            timeout=3600,
        )
        final_blob = bucket.blob(blob_name)
        _rewrite_gcs_blob(temp_blob, final_blob)
        temp_blob.delete(timeout=3600)
    else:
        parent = os.path.dirname(path)
        if parent:
            gfile.makedirs(parent)
        path_tmp = path + "-TEMPORARY"
        with gfile.GFile(path_tmp, "wb") as f:
            io_buffer.seek(0)
            while True:
                chunk = io_buffer.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        gfile.rename(path_tmp, path, overwrite=True)

    if step_copy is not None:
        step_path = f"{path}-{step_copy:09d}"
        if _is_gs_path(path):
            from google.cloud import storage

            src_bucket_name, src_blob_name = _parse_gs_path(path)
            dst_bucket_name, dst_blob_name = _parse_gs_path(step_path)
            client = storage.Client()
            src_bucket = client.bucket(src_bucket_name)
            dst_bucket = client.bucket(dst_bucket_name)
            src_blob = src_bucket.blob(src_blob_name)
            dst_blob = dst_bucket.blob(dst_blob_name)
            _rewrite_gcs_blob(src_blob, dst_blob)
        else:
            gfile.copy(path, step_path, overwrite=True)


def recover_tree(keys, values):
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree


def _sync(x):
    return jax.lax.psum(x, "i")


def sync():
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(_sync, "i")(x))
    assert x[0] == jax.device_count()


def check_and_compile_patterns(patterns):
    if isinstance(patterns, str):
        patterns = [patterns]

    assert isinstance(patterns, (list, tuple)), patterns

    def check_and_compile(pattern):
        assert not pattern.startswith("/"), (
            f"Big vision parameter names never start with '/': '{pattern}")
        return re.compile(pattern)

    return list(map(check_and_compile, patterns))


def make_mask_trees(tree, patterns, *, log=None):
    compiled_patterns = check_and_compile_patterns(patterns)

    def matchfirst(name, _):
        matches = []
        for pattern in compiled_patterns:
            matches.append(not any(matches) and bool(pattern.fullmatch(name)))
        if log is not None and True in matches and jax.process_index() == 0:
            logging.info("%s: %s - matched by %s", log, name,
                            patterns[matches.index(True)])
        return np.array(matches)

    multimask = tree_map_with_names(matchfirst, tree)
    return [
        jax.tree_util.tree_map(lambda matches, i=idx: matches[i], multimask)
        for idx in range(len(patterns))
    ]


class MetricWriter:

    def __init__(self, workdir=None, config=None):
        self.step_start(0)
        if jax.process_index() != 0: return

        self.pool = multiprocessing.pool.ThreadPool(1)
        self.fname = None
        if workdir:
            self.fname = os.path.join(workdir, "big_vision_metrics.txt")
            if config:
                with gfile.GFile(os.path.join(workdir, "config.json"), "w") as f:
                    f.write(config.to_json())

    def step_start(self, step):
        self.step = step
        self.step_metrics = {}

    def measure(self, name, value):
        if jax.process_index() != 0: return

        value = np.array(value).squeeze()

        value = float(value) if value.ndim == 0 else value.shape

        logging.info(f"\u001b[35m[{self.step}]\u001b[0m {name} = {value}")
        logging.flush()
        self.step_metrics[name] = value

        return value

    def step_end(self):
        if not self.step_metrics: return

        def write(metrics):
            with gfile.GFile(self.fname, "a") as f:
                f.write(json.dumps({"step": self.step, **metrics}) + "\n")

        if self.fname:
            self.pool.apply(lambda: None)
            self.pool.apply_async(write, (self.step_metrics,))

    def close(self):
        self.step_end()
        if jax.process_index() == 0:
            self.pool.close()
            self.pool.join()

import collections
import functools
import itertools
from typing import Sequence

import datasets.build_transforms as pp_builder
import datasets.tfds as ds_tfds
from gemma.gm import data as gemma_data
import jax
import tensorflow as tf
import numpy as np


def build_single_source_train_dataset(
    data,
    preprocess_fn,
    shuffle_buffer_size,
    filter_fn=None,
    num_parallel_calls=100,
):
    data = _add_tpu_host_options(data)
    if filter_fn:
        data = data.filter(filter_fn)
    data = data.repeat(None)  # repeat the dataset indefinitely
    data = data.shuffle(shuffle_buffer_size) if shuffle_buffer_size else data
    data = data.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
    return data.prefetch(2)


def build_training_dataset(input_config):
    if not isinstance(input_config.data, (list, tuple)):
        raise TypeError(
            "input_config.data must be a list of (dataset_cfg, weight) pairs. "
            f"Got {type(input_config.data).__name__}."
        )

    datasets = []
    weights = []
    ntraining_examples = 0

    for dataset_cfg, weight in input_config.data:
        subset_count = None
        try:
            subset_count = dataset_cfg.get("subset_count")
        except Exception:
            subset_count = None
        data_kw = dict(dataset_cfg)
        data_kw.pop("subset_count", None)

        train_data = ds_tfds.DataSource(**data_kw)
        filter_fn = input_config.get("filter_fn")

        subset_fn = None
        if subset_count:
            subset_fraction = min(
                1.0, float(subset_count) / float(train_data.total_examples)
            )
            subset_fn = _build_tfds_id_subsample_filter(subset_fraction, 0)
            effective_examples = min(int(subset_count), train_data.total_examples)
        else:
            effective_examples = train_data.total_examples
        filter_fn = _combine_filters(filter_fn, subset_fn)
        ntraining_examples += effective_examples
        dataset = build_single_source_train_dataset(
            data=train_data.get_tfdata(),
            preprocess_fn=pp_builder.get_preprocess_fn(input_config.preprocess),
            shuffle_buffer_size=int(input_config.shuffle_buffer_size * weight),
            filter_fn=filter_fn,
        )
        datasets.append(dataset)
        weights.append(float(weight))
    weight_sum = sum(weights)
    weights = [x / weight_sum for x in weights]
    train_ds = tf.data.Dataset.sample_from_datasets(
        datasets, weights, stop_on_empty_dataset=True
    )
    train_ds = train_ds.batch(
        input_config["batch_size"] // jax.process_count(), drop_remainder=True
    )
    return train_ds, ntraining_examples


def _add_tpu_host_options(data):
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    return data.with_options(options)


def _build_tfds_id_subsample_filter(subset_fraction, seed=0):
    if subset_fraction is None:
        return None
    assert (subset_fraction > 0) and (subset_fraction <= 1), subset_fraction
    num_buckets = 2**31 - 1
    threshold = int(num_buckets * subset_fraction)
    salt = tf.constant(str(seed))

    def _filter(example):
        tfds_id = example.get("tfds_id")
        if tfds_id is None:
            return tf.constant(False)
        key = tf.strings.join([tfds_id, salt])
        bucket = tf.strings.to_hash_bucket_fast(key, num_buckets)
        return bucket < threshold

    return _filter


def _combine_filters(*filters):
    filters = [f for f in filters if f is not None]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]

    def _combined(example, _filters=filters):
        keep = _filters[0](example)
        for fn in _filters[1:]:
            keep = tf.logical_and(keep, fn(example))
        return keep

    return _combined


def prefetch_iterator(it, n):
    if not n:
        yield from it
        return
    queue = collections.deque()

    def enqueue(n_steps):  # Enqueues *up to* `n` elements from the iterator.
        for item in itertools.islice(it, n_steps):
            queue.append(item)

    enqueue(n)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)

"""
Helper for T5 Gemma tokenizer
"""
def _is_str_array(x) -> bool:
    if not isinstance(x, np.ndarray):
        return False
    return np.dtype(x.dtype).type in {np.object_, np.str_}


def _normalize_prompt(prompt: str | Sequence[str]) -> list[str]:
    """Normalize the inputs."""
    if _is_str_array(prompt):  # Supports batched input array
        assert isinstance(prompt, np.ndarray)
        prompt = prompt.tolist()

    return [prompt] if isinstance(prompt, str) else list(prompt)


def tokenize(x, tokenizer=None):
    """
    In this pipeline, images and texts are processed separately, since jax.tree_util.tree_map applies this function to each batch leaf.
    x can be image tensor or text tensor. They're distinguished based on tensor shape. Concretely:
    - tokenize(image_tensor, ...) -> no op
    - tokenize(text_tensor, ...) -> tokenize the text sequence
    """
    x = x._numpy()

    def _encode_with_tokenizer(tok, arr):
        if hasattr(tok, "_encode_prompts"):
            prompt = _normalize_prompt([_.decode() for _ in arr.tolist()])
            tokens = [tok.tokenizer.encode(p)[: tok.max_input_length] for p in prompt]
            temp = np.asarray(
                gemma_data.pad(tokens, max_length=tok.max_input_length), dtype=np.int32
            )
            mask = (temp != 0).astype(np.int32)  # PAD_ID is 0
            return np.stack([temp, mask], axis=-1)
        else:
            """
            For text tokenizer from the transformers library:
            maps, e.g., (bs=16) to (bs=16, seqlen=77, 2)
            2 is input_ids (in the vocabulary) + attention_mask
            temp['input_ids'],temp['attention_mask'] are both of dimensions (16, 77)
            """
            temp = tok([_.decode() for _ in arr.tolist()], return_tensors="np")
            return np.transpose(
                np.array(
                    [
                        temp["input_ids"],
                        temp["attention_mask"]
                        if temp.get("attention_mask") is not None
                        else np.ones_like(temp["input_ids"]),
                    ]
                ),
                (1, 2, 0),
            )

    if tokenizer is not None and len(x.shape) == 1:
        if isinstance(tokenizer, (list, tuple)):
            return [_encode_with_tokenizer(tok, x) for tok in tokenizer]
        else:
            return _encode_with_tokenizer(tokenizer, x)
    return x


def start_tokenize_input_iterator(data, n_prefetch=1, tokenizer=None):
    fn = functools.partial(tokenize, tokenizer=tokenizer)
    it = (jax.tree_util.tree_map(fn, elem) for elem in iter(data))
    return prefetch_iterator(it, n_prefetch)

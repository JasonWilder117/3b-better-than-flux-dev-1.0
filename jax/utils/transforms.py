"""
Data preprocessing operations.
Data is represented as a dictionary of TensorFlow tensors. Each operation returns a modified dictionary.
"""

from utils.registry import Registry, InKeyOutKey
import jax
import tensorflow as tf


@Registry.register("preprocess_ops.value_range")
@InKeyOutKey()
def get_value_range(vmin=-1, vmax=1, in_min=0, in_max=255.0, clip_values=False):
    """
    Transforms a [in_min,in_max] image to [vmin,vmax] range.
    """

    def _value_range(image):
        """Scales values in given range."""
        in_min_t = tf.constant(in_min, tf.float32)
        in_max_t = tf.constant(in_max, tf.float32)
        image = tf.cast(image, tf.float32)
        image = (image - in_min_t) / (in_max_t - in_min_t)
        image = vmin + image * (vmax - vmin)
        if clip_values:
            image = tf.clip_by_value(image, vmin, vmax)
        return image

    return _value_range


@Registry.register("preprocess_ops.decode_png")
@InKeyOutKey(with_data=True)
def get_decode_png(channels=3, out_type="uint8", shape_key="image/shape"):
    """
    Decode PNG-encoded bytes into a tensor, using shape metadata when available.
    """
    out_type = tf.as_dtype(out_type)

    def _decode_png(image_bytes, data):
        if image_bytes.dtype != tf.string:
            return tf.cast(image_bytes, out_type)
        image = tf.io.decode_png(image_bytes, channels=channels)
        if image.dtype != out_type:
            image = tf.cast(image, out_type)
        shape = data.get(shape_key)
        if shape is not None:
            image = tf.reshape(image, tf.cast(shape, tf.int32))
        return image

    return _decode_png


@Registry.register("preprocess_ops.keep")
def get_keep(*keys):
    """
    Keeps only the given keys.
    """

    def _keep(data):
        return {k: v for k, v in data.items() if k in keys}

    return _keep


@Registry.register("preprocess_ops.copy")
def get_copy(inkey, outkey):
    """
    Copies value of `inkey` into `outkey`.
    """

    def _copy(data):
        data[outkey] = jax.tree_util.tree_map(lambda x: x, data[inkey])
        return data

    return _copy


@Registry.register("preprocess_ops.sample_caption")
@InKeyOutKey()
def get_sample_caption():
    """
    - If input is a list of captions: randomly select one
    - If input is a string: no-op
    - If input is an empty list: output an empty string
    """

    def _sample_caption(x):
        if isinstance(x, tf.RaggedTensor):
            x = x.flat_values
        x = tf.reshape(x, [-1])
        n = tf.shape(x)[0]

        def _pick():
            idx = tf.random.uniform([], maxval=n, dtype=tf.int32)
            return x[idx]

        return tf.cond(n > 0, _pick, lambda: tf.constant("", dtype=x.dtype))

    return _sample_caption


@Registry.register("preprocess_ops.first_caption")
@InKeyOutKey()
def get_first_caption():
    """
    - If input is a list of captions: select the first one
    - If input is a string: no-op
    - If input is an empty list: output an empty string
    """

    def _first_caption(x):
        if isinstance(x, tf.RaggedTensor):
            x = x.flat_values
        x = tf.reshape(x, [-1])
        n = tf.shape(x)[0]

        def _pick():
            return x[0]

        return tf.cond(n > 0, _pick, lambda: tf.constant("", dtype=x.dtype))

    return _first_caption

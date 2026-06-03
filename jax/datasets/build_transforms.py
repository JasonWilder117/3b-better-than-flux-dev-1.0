from absl import logging
from utils.registry import Registry
import utils.transforms


def get_preprocess_fn(pp_pipeline):
    ops = []
    if pp_pipeline:
        for fn_name in pp_pipeline.split("|"):
            if not fn_name:
                continue
            try:
                ops.append(Registry.lookup(f"preprocess_ops.{fn_name}")())
            except SyntaxError as err:
                raise ValueError(f"Syntax error on: {fn_name}") from err

    def _preprocess_fn(data):
        logging.info("Data before pre-processing:\n%s", data)
        for op in ops:
            data = op(data)
        if not isinstance(data, dict):
            raise ValueError("Argument `data` must be a dictionary, not %s" % str(type(data)))
        logging.info("Data after pre-processing:\n%s", data)
        return data
    return _preprocess_fn

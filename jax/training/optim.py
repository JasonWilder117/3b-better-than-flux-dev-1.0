import operator
import jax
import optax
import utils.common as u

def steps(prefix, config, data_size=None, batch_size=None, total_steps=None,
            default=ValueError):
    suffixes = {"steps", "examples", "epochs", "percent"}
    matches = {f"{prefix}_{s}" for s in suffixes if f"{prefix}_{s}" in config}
    assert len(matches) <= 1, f"Only one of '{matches}' should be defined."

    if f"{prefix}_steps" in config:
        return config[f"{prefix}_steps"]

    if batch_size and f"{prefix}_examples" in config:
        return max(round(config[f"{prefix}_examples"] / batch_size), 1)

    if batch_size and data_size and f"{prefix}_epochs" in config:
        steps_per_epoch = data_size / batch_size
        return max(round(config[f"{prefix}_epochs"] * steps_per_epoch), 1)

    if total_steps and f"{prefix}_percent" in config:
        pct = config[f"{prefix}_percent"]
        assert 0.0 <= pct <= 1.0, (
            f"Percents should lie in [0.0, 1.0], but {prefix}_percent is {pct}")
        return max(round(pct * total_steps), 1)

    if default is ValueError:
        raise ValueError(
            f"Cannot convert {prefix} to steps, due to missing batch_size "
            f"({batch_size}), data_size ({data_size}), total_steps ({total_steps})"
            ", or corresponding entry in config:\n" + "\n".join(config.keys()))

    return default


def find_states(opt_state, cls):
    leaves = jax.tree_util.tree_leaves(
        opt_state, is_leaf=lambda node: isinstance(node, cls))
    return [leaf for leaf in leaves if isinstance(leaf, cls)]


def get_count(opt_state):
    counts = {
        int(state.count)
        for state in find_states(opt_state, optax.ScaleByScheduleState)
    }
    assert len(counts) == 1, f"Expected exactly 1 ScaleByScheduleState: {counts}"
    return next(iter(counts))


def replace_frozen(schedule, pytree, replacement, log=None):
    if not isinstance(schedule, (list, tuple)):
        return pytree
    masks, scheds = _make_mask_trees(pytree, schedule, log=log)
    frozen_mask, _, _ = _split_frozen(masks, scheds)
    return jax.tree_util.tree_map(
        lambda v, f: replacement if f else v, pytree, frozen_mask)


def make(config, params):

    schedule = config.schedule
    if not isinstance(schedule, (tuple, list)):
        schedule = [(".*", schedule)]
    masks, scheds = _make_mask_trees(params, schedule, "config.schedule")
    frozen_mask, masks, scheds = _split_frozen(masks, scheds)
    not_frozen_mask = jax.tree_util.tree_map(operator.not_, frozen_mask)
    schedule_fns = [optax.constant_schedule(mult) for mult in scheds]
    schedule_txs = [
        optax.masked(optax.scale_by_schedule(schedule_fn), mask)
        for schedule_fn, mask in zip(schedule_fns, masks)
    ] + [
        optax.masked(optax.set_to_zero(), frozen_mask)
    ]

    grad_clip_norm_tx = (
        optax.masked(optax.clip_by_global_norm(config.grad_clip_norm),
                        not_frozen_mask)
        if config.get("grad_clip_norm") else optax.identity())

    tx_func = operator.attrgetter(config.optax_name)(optax)
    opt_txs = [optax.masked(tx_func(**config.get("optax", {})), not_frozen_mask)]
    assert "optim" not in config, "Deprecated option, use config.optax."

    lr_mult_txs = [optax.scale(config.lr)]
    if config.get("lr_mults"):
        masks, mults = _make_mask_trees(params, config.lr_mults, "config.lr_mults")
        assert all(mult > 0 for mult in mults), (
            f"Use schedule=None for parameter freezing instead of lr_mults={mults}")
        lr_mult_txs += [
            optax.masked(optax.scale(mult), mask)
            for mult, mask in zip(mults, masks)
        ]

    assert "weight_decay" not in config, "Deprecated option. Use wd and schedule."
    assert config.get("weight_decay_decouple", True), (
        "Coupled weight decay not supported anymore.")
    if config.get("wd"):
        wd_mults = config.get("wd_mults", [(".*/kernel$", 1.0)])
        masks, mults = _make_mask_trees(params, wd_mults, "config.wd_mults")
        weight_decay_txs = [
            optax.add_decayed_weights(config.wd * mult, mask)
            for mult, mask in zip(mults, masks)
        ]
    else:
        weight_decay_txs = []
    return optax.chain(
        grad_clip_norm_tx,
        *opt_txs,
        *weight_decay_txs,
        *lr_mult_txs,
        *schedule_txs,
        optax.scale(-1.0)), schedule_fns


def _make_mask_trees(params, patterns_values, log):
    patterns, values = zip(*patterns_values)
    masks = u.make_mask_trees(params, patterns, log=log)
    return masks, values


def _split_frozen(masks, scheds):
    all_false = jax.tree_util.tree_map(lambda *bools: not any(bools), *masks)
    assert not any(jax.tree_util.tree_flatten(all_false)[0]), (
        f"All params must be covered (use `None` for freezing): {all_false}")
    frozen_masks = [
        mask for mask, sched in zip(masks, scheds) if sched is None]
    frozen_mask = jax.tree_util.tree_map(
        lambda *bools: any(bools), *frozen_masks,
        all_false)
    masks, scheds = zip(*(
        (mask, sched) for mask, sched in zip(masks, scheds) if sched is not None))
    return frozen_mask, masks, scheds

"""Learning rate schedulers built from a model's ``scheduler_config`` (see ``build_scheduler``).

``scheduler_config`` is a plain dict (yaml friendly) given to the model constructor (or to
``TimeSeries.set_model(config=dict(scheduler_config=...))``):

- ``None`` (default): no scheduler, the learning rate stays ``optim_config['lr']``.
- a dict WITHOUT ``type`` (or ``type: step``): ``torch.optim.lr_scheduler.StepLR`` stepped once per EPOCH with the
  remaining keys, e.g. ``{step_size: 10, gamma: 0.5}`` (the behaviour of dsipts < 1.10, unchanged).
- ``type: cosine``: stepped after every OPTIMIZER STEP (batch). Optional linear warmup from 0 to ``lr`` over
  ``warmup_steps``, then a half cosine from ``lr`` down to ``min_lr_ratio * lr`` over the remaining
  ``total_steps - warmup_steps`` steps, then constant at ``min_lr_ratio * lr``::

      scheduler_config:
        type: cosine
        total_steps: 566000   # REQUIRED: number of optimizer steps the decay spans, or 'auto'
        min_lr_ratio: 0.1     # final lr = 0.1 * lr (default 0.1)
        warmup_steps: 0       # linear warmup steps (default 0)

  ``total_steps`` must be given explicitly: the steps per epoch are ``len(train_dataset) // batch_size`` (training
  logs ``train:<steps per epoch>, validation:<...>`` at the start), so for E epochs ``total_steps = E * steps``.
  ``total_steps: auto`` uses Lightning's ``trainer.estimated_stepping_batches`` (= max_epochs x steps per epoch):
  the decay then spans ``max_epochs``, NOT the epoch where early stopping ends the training, so with a large
  max_epochs and early stopping the learning rate barely decays.

  With ``warmup_steps: 0`` and ``min_lr_ratio: 0.1`` this is the schedule of s4casting (Alliander) training: cosine
  from lr to lr/10 over ``maximum_steps``, then lr/10.

The lr at each epoch's validation is written in the ``lr`` column of ``loss.csv`` (MetricsCallback).
Not supported: a per-step scheduler with the SAM optimizer (manual optimization, Lightning does not step it).
"""
import math
from typing import Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, StepLR

SCHEDULER_TYPES = ('step', 'cosine')
_COSINE_KEYS = {'type', 'total_steps', 'min_lr_ratio', 'warmup_steps'}


def cosine_factor(step: int, total_steps: int, min_lr_ratio: float = 0.1, warmup_steps: int = 0) -> float:
    """Multiplier of the base lr after ``step`` optimizer steps (warmup, half cosine, then constant).

    Args:
        step (int): optimizer steps done so far (0 = before the first step)
        total_steps (int): steps at the end of the decay (warmup included)
        min_lr_ratio (float): final lr / base lr
        warmup_steps (int): linear warmup steps (the first step uses lr / warmup_steps)

    Returns:
        float: the multiplier, in (0, 1]
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr_ratio + 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - min_lr_ratio)


def _check_cosine(cfg: dict) -> None:
    unknown = set(cfg) - _COSINE_KEYS
    assert not unknown, f'scheduler_config type cosine: unknown keys {sorted(unknown)} (allowed: {sorted(_COSINE_KEYS)})'
    total = cfg.get('total_steps')
    assert total is not None, ('scheduler_config type cosine needs total_steps (optimizer steps the decay spans: '
                               'epochs x len(train) // batch_size), or total_steps: auto for max_epochs x steps per epoch')
    assert total == 'auto' or (isinstance(total, int) and total > 0), f'total_steps must be a positive int or "auto", got {total!r}'
    ratio = cfg.get('min_lr_ratio', 0.1)
    assert 0.0 <= ratio <= 1.0, f'min_lr_ratio must be in [0, 1], got {ratio}'
    warmup = cfg.get('warmup_steps', 0)
    assert isinstance(warmup, int) and warmup >= 0, f'warmup_steps must be a non-negative int, got {warmup!r}'
    if isinstance(total, int):
        assert warmup < total, f'warmup_steps ({warmup}) must be smaller than total_steps ({total})'


def build_scheduler(optimizer: Optimizer, scheduler_config: Union[dict, None], trainer=None) -> Union[dict, None]:
    """The Lightning ``lr_scheduler`` dict for ``optimizer`` (or None), from ``scheduler_config`` (see module doc).

    Args:
        optimizer (Optimizer): the optimizer whose lr is scheduled
        scheduler_config (Union[dict, None]): None, a StepLR config, or ``{type: cosine, total_steps, ...}``
        trainer (optional): the Lightning trainer, only needed with ``total_steps: auto``

    Returns:
        Union[dict, None]: ``{'scheduler', 'interval', 'frequency'}`` to put under ``lr_scheduler`` in
        ``configure_optimizers``, or None (no scheduler)
    """
    if scheduler_config is None:
        return None
    cfg = dict(scheduler_config)
    kind = cfg.pop('type', 'step')
    assert kind in SCHEDULER_TYPES, f'scheduler_config type must be one of {SCHEDULER_TYPES}, got {kind!r}'
    if kind == 'step':
        return {'scheduler': StepLR(optimizer, **cfg), 'interval': 'epoch', 'frequency': 1}

    _check_cosine({'type': kind, **cfg})
    total = cfg['total_steps']
    if total == 'auto':
        assert trainer is not None, 'total_steps: auto needs the Lightning trainer'
        total = int(trainer.estimated_stepping_batches)
    ratio = float(cfg.get('min_lr_ratio', 0.1))
    warmup = int(cfg.get('warmup_steps', 0))
    scheduler = LambdaLR(optimizer, lambda step: cosine_factor(step, total, ratio, warmup))
    return {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}


def is_per_step(scheduler_config: Union[dict, None]) -> bool:
    """True if ``scheduler_config`` is stepped after every optimizer step (not once per epoch)."""
    return scheduler_config is not None and dict(scheduler_config).get('type', 'step') != 'step'

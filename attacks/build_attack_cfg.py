"""Factory: build an AttackConfig from a DatasetBundle."""
from __future__ import annotations

import numpy as np

from attacks.base import AttackConfig
from data.pipeline import DatasetBundle
from data.schema import FEATURE_SPECS


def build_attack_config(bundle: DatasetBundle, epsilon: float = 0.10) -> AttackConfig:
    """Build an AttackConfig aligned with the preprocessor's feature layout."""
    pre = bundle.preprocessor
    names = pre.feature_names_
    integer_mask = np.array([FEATURE_SPECS[n].integer if n in FEATURE_SPECS else False
                             for n in names])
    return AttackConfig(
        epsilon=epsilon,
        clip_lower=pre.lower_,
        clip_upper=pre.upper_,
        actionable_mask=pre.actionable_mask(),
        direction=pre.direction_array(),
        integer_mask=integer_mask,
    )

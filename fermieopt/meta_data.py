from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np


class EffectMetaData(TypedDict):
    fixed_effects: np.ndarray[Union[int, float]]
    specific_effects: np.ndarray[Union[int, float]]


class InvestMetaData(TypedDict):
    costs: EffectMetaData
    funding: EffectMetaData


class MetaData(TypedDict):
    invest: InvestMetaData
    Gruppe: Optional[str]


class MetaDataFactory:
    length = 1

    @classmethod
    def create(cls) -> MetaData:
        return {
            'invest': {
                'costs': {
                    'fixed_effects': np.array([0] * cls.length, dtype=float),
                    'specific_effects': np.array([0] * cls.length, dtype=float),
                },
                'funding': {
                    'fixed_effects': np.array([0] * cls.length, dtype=float),
                    'specific_effects': np.array([0] * cls.length, dtype=float),
                },
            },
            'Gruppe': None,
        }

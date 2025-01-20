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


class MetaDataFactory:
    @staticmethod
    def create() -> MetaData:
        return {
            'invest': {
                'costs': {
                    'fixed_effects': np.array([0], dtype=float),
                    'specific_effects': np.array([0], dtype=float),
                },
                'funding': {
                    'fixed_effects': np.array([0], dtype=float),
                    'specific_effects': np.array([0], dtype=float),
                },
            },
        }

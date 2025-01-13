from typing import Optional, Union, Any, Dict, Literal, List, Tuple, TypedDict

import numpy as np


class EffectMetaData(TypedDict):
    fixed_effects: Dict[np.ndarray[Union[int, float]]]
    specific_effects: Dict[np.ndarray[Union[int, float]]]


class InvestMetaData(TypedDict):
    costs: EffectMetaData
    funding: EffectMetaData


class MetaData(TypedDict):
    invest: InvestMetaData


class MetaDataFactory:
    @staticmethod
    def create() -> MetaData:
        return {
            "invest": {
                "costs": {
                    "fixed_effects": np.array([0], dtype=float),
                    "specific_effects": np.array([0], dtype=float),
                },
                "funding": {
                    "fixed_effects": np.array([0], dtype=float),
                    "specific_effects": np.array([0], dtype=float),
                },
            },
        }

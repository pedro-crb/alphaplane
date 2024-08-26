from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from alphaplane.analysis_tools.analysis_condition import AnalysisCondition


class PropellerCondition(AnalysisCondition):
    def __init__(self, J: float | np.ndarray | list[float] | None = None,
                 rpm: float | np.ndarray | list[float] | None = None,
                 is_sequence: bool = False,
                 ) -> None:
        super().__init__()
        self.J: np.ndarray | None = np.atleast_1d(J) if J is not None else None
        self.rpm: np.ndarray | None = np.atleast_1d(rpm) if rpm is not None else None
        self._is_sequence: bool = is_sequence

    def set_defaults(self) -> None:
        default_values = {
            'J': 0.0,
            'rpm': 5000.0,
        }
        for attr, default in default_values.items():
            value = getattr(self, attr, None)
            if (value is None) or (len(value) == 0):
                setattr(self, attr, np.atleast_1d(default))
                self._invalidate_cache()

    @property
    def is_single_point(self) -> bool:
        assert self.J is not None and self.rpm is not None
        conditions = [
            isinstance(self.J, float) or len(np.array(self.J)) == 1,
            isinstance(self.rpm, float) or len(np.array(self.rpm)) == 1,
        ]
        return all(conditions)

    @classmethod
    def num_params(cls) -> int:
        return 2

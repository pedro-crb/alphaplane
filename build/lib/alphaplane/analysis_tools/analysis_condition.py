from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from alphaplane.numerical_tools.array_operations import cartesian_product


class AnalysisCondition(ABC):
    def __init__(self) -> None:
        self._is_sequence: bool = False
        self._can_be_sequence_cache = None
        self._parameter_grid_cache = None
        self._parameter_list_cache = None
        self._parameter_sequence_cache = None

        if self._is_sequence:
            assert self.can_be_sequence, 'Inputs do not form a sequence'

    def __hash__(self):
        attribute_names = self.__dict__.keys()
        attribute_hashes = (getattr(self, name).tobytes() if getattr(self, name) is not None else None
                            for name in attribute_names if not name.startswith('_'))
        return hash(tuple(attribute_hashes))

    def __eq__(self, other):
        if not isinstance(other, AnalysisCondition):
            raise NotImplementedError
        checks = [np.array_equal(value, getattr(other, key)) for key, value in 
                  self.__dict__.items() if not key.startswith('_')]
        return all(checks) and not self.is_sequence^other.is_sequence

    @abstractmethod
    def set_defaults(self) -> None:
        pass

    def fill_with_other(self, other: AnalysisCondition) -> None:
        for key, value in self.__dict__.items():
            if key != 'is_sequence' and not key.startswith('_'):
                other_value = getattr(other, key)
                if value is None or len(value) == 0:
                    setattr(self, key, other_value)

    @property
    def is_sequence(self) -> bool:
        return self._is_sequence

    @is_sequence.setter
    def is_sequence(self, value) -> None:
        self._is_sequence = value

    @property
    def can_be_sequence(self) -> bool:
        if self._can_be_sequence_cache is None:
            param_list = self.parameter_list()
            lengths = [len(p) for p in param_list if len(p) > 1]
            self._can_be_sequence_cache = all([length == lengths[0] for length in lengths])
        return self._can_be_sequence_cache

    def parameter_grid(self) -> np.ndarray:
        if self._parameter_grid_cache is None:
            parameters = [
                np.atleast_1d(getattr(self, key))
                for key in self.__dict__.keys()
                if not key.startswith('_')
            ]
            self._parameter_grid_cache = cartesian_product(*parameters)

        return self._parameter_grid_cache

    def parameter_list(self) -> list[np.ndarray]:
        if self._parameter_list_cache is None:
            self._parameter_list_cache = [
                (np.atleast_1d(np.array(getattr(self, key), dtype=float))
                 if getattr(self, key) is not None else np.array([]))
                for key in self.__dict__.keys()
                if not key.startswith('_')
            ]
        return self._parameter_list_cache

    def parameter_sequence(self):
        if self._parameter_sequence_cache is None:
            assert self.can_be_sequence, 'Condition does not form a sequence'
            param_list = self.parameter_list()
            max_len = max(len(item) for item in param_list)
            new_parameter_list = [p if len(p) > 0 else [[]] for p in param_list]
            final_conditions = [[item[0]] * max_len if len(item) == 1 else item for item in new_parameter_list]
            self._parameter_sequence_cache = np.array(final_conditions, dtype='object').T
        return self._parameter_sequence_cache

    def _invalidate_cache(self):
        self._can_be_sequence_cache = None
        self._parameter_grid_cache = None
        self._parameter_list_cache = None
        self._parameter_sequence_cache = None

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if not key.startswith('_'):
            self._invalidate_cache()
            
    @property
    def var_names(self) -> list[str]:
        return [key for key in self.__dict__.keys() if not key.startswith('_')]

    @classmethod
    @abstractmethod
    def num_params(cls) -> int:
        raise NotImplementedError

    @classmethod
    def from_intersection(cls, conditions: list[Self | None]):
        mins = []
        maxs = []
        nums = []
        for cond in conditions:
            mins.append([min(lst) for lst in cond.parameter_list() if len(lst) > 0])
            maxs.append([max(lst) for lst in cond.parameter_list() if len(lst) > 0])
            nums.append([len(lst) for lst in cond.parameter_list() if len(lst) > 0])
        mins = np.array(mins)
        maxs = np.array(maxs)
        nums = np.array(nums)
        size = maxs - mins
        size = np.where(size == 0, 1, size)
        dens = (nums - 1) / size
        min_final = np.max(mins, axis=0)
        max_final = np.min(maxs, axis=0)
        dens_final = np.max(dens, axis=0)
        condition_values = [np.linspace(mn, mx, int(np.ceil((mx - mn) * d)) + 1)
                            for mn, mx, d in zip(min_final, max_final, dens_final)]
        return cls(*condition_values)

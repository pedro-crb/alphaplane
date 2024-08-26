from __future__ import annotations

from typing import Any

from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

import numpy as np
import copy

from alphaplane.numerical_tools.array_operations import cartesian_product, search_sorted_with_nan, ndgrad
from alphaplane.analysis_tools.analysis_condition import AnalysisCondition


class AnalysisData:
    def __init__(self, condition_type, output_var_names, allow_interpolation: bool = True):
        self._condition_type = condition_type
        self._input_var_names = condition_type().var_names
        self._output_var_names = output_var_names
        self._num_inputs = condition_type.num_params()
        self._num_outputs = len(self._output_var_names)
        self._allow_interpolation = allow_interpolation

        self._condition_structured: condition_type | None = None
        self._condition_unstructured: condition_type | None = None
        self._data_structured: np.ndarray | None = None
        self._data_ndpolator: RegularGridInterpolator | None = None
        self._data_unstructured: np.ndarray | None = None
        self._data_all: np.ndarray | None = None
        self._interpolator_cache: dict[str, Any] = {
            'hash': None,
            'interpolator': None,
            'indices': None,
            'uniques': None,
        }
        self._gradient_cache: dict[str, int | list] = \
            {'hash': 0, 'grads': [np.array(0.0)] * self.num_inputs}

    def __hash__(self):
        return hash((
            self._condition_structured,
            self._data_structured.tobytes() if self._data_structured is not None else None,
            self._data_unstructured.tobytes() if self._data_unstructured is not None else None,
            self._data_all.tobytes() if self._data_all is not None else None,
        ))

    @property
    def input_var_names(self):
        return self._input_var_names

    @property
    def output_var_names(self):
        return self._output_var_names

    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def num_outputs(self):
        return self._num_outputs

    @property
    def all_var_names(self):
        return self._input_var_names + self._output_var_names

    @property
    def condition_structured(self):
        return copy.deepcopy(self._condition_structured)

    @property
    def data_structured(self) -> np.ndarray:
        assert self._data_structured is not None
        return self._data_structured

    @property
    def data_unstructured(self) -> np.ndarray:
        assert self._data_unstructured is not None
        return self._data_unstructured

    @property
    def condition_unstructured(self) -> AnalysisCondition:
        _, _, uniques = self._build_unstructured_interpolator()
        return self._condition_type(*uniques)

    @property
    def has_unstructured(self) -> bool:
        return self._data_unstructured is not None

    @property
    def condition(self):
        return self.condition_unstructured if self.has_unstructured else self.condition_structured

    def get_all_data(self) -> np.ndarray:
        if self._data_all is None:
            self._data_all = self.get_points()
        assert self._data_all is not None
        return self._data_all

    def interpolate_structured(self, query: np.ndarray) -> np.ndarray:
        assert self._allow_interpolation
        if self._data_ndpolator is None:
            raise ValueError('No data to interpolate')
        result = self._data_ndpolator(np.atleast_2d(query))
        result = np.hstack((query, result))
        return result

    def interpolate_general(self, query: np.ndarray) -> np.ndarray:
        assert self._allow_interpolation
        interpolator, valid_dims, uniques = self._build_unstructured_interpolator()
        data = self.get_all_data()
        valid_indices = [i for (i, qu) in enumerate(query) if
                         all([np.isin(q, u) for (j, (q, u)) in enumerate(zip(qu, uniques)) if j not in valid_dims])]
        result = np.full((query.shape[0], data.shape[1]), np.nan)

        query_filtered = query[valid_indices][:, valid_dims]
        interpolation_result = interpolator(query_filtered)
        result[valid_indices] = interpolation_result
        return interpolation_result

    def _build_unstructured_interpolator(self) -> tuple[LinearNDInterpolator, list[int], list[np.ndarray]]:
        if self._interpolator_cache['hash'] == hash(self):
            interp = self._interpolator_cache
            return interp['interpolator'], interp['indices'], interp['uniques']
        data = np.atleast_2d(self.get_all_data())
        independent_vars = data[:, :self._condition_type.num_params()]
        dependent_vars = data
        uniques = [np.unique(v) for v in independent_vars.T]
        indices = [i for (i, u) in enumerate(uniques) if len(u) > 1]
        independent_vars = independent_vars[:, indices]
        interpolator = LinearNDInterpolator(independent_vars, dependent_vars, rescale=True)
        self._interpolator_cache['hash'] = hash(self)
        self._interpolator_cache['interpolator'] = interpolator
        self._interpolator_cache['indices'] = indices
        self._interpolator_cache['uniques'] = uniques
        return interpolator, indices, uniques

    def get_points(
            self, condition: AnalysisCondition | None = None, datasets: str = 'all'
    ) -> np.ndarray:
        condition = condition if condition is not None else self._condition_type()
        if condition.is_sequence:
            sequence = condition.parameter_sequence()
            conditions = [self._condition_type(*v) for v in sequence]
        else:
            conditions = [condition]

        if datasets == 'all' or datasets == 'structured':
            structured_data = [self.get_points_structured(c) for c in conditions]
            structured_polar = np.concatenate(structured_data)
        else:
            structured_polar = np.array([])

        if datasets == 'all' or datasets == 'unstructured':
            unstruct_polar = self.get_points_unstructured(condition)
        else:
            unstruct_polar = np.array([])

        polars = [p for p in [structured_polar, unstruct_polar] if p is not None and len(p) != 0]

        if len(polars) == 0:
            return np.array([])

        data = np.concatenate(polars)
        return data

    def get_points_unstructured(self, condition: AnalysisCondition | None = None
                                ) -> np.ndarray:
        condition = condition if condition is not None else self._condition_type()
        param_list = condition.parameter_list()
        data = []
        check_indices = [i for i, p in enumerate(param_list) if len(p) > 0]

        if self._data_unstructured is None:
            return np.array([])
        if len(check_indices) == 0:
            return self._data_unstructured

        param_list_filtered = [param_list[i] for i in check_indices]
        if condition.is_sequence:
            assert condition.can_be_sequence, ('To get points from a sequence, all parameters '
                                               'given in condition must have the same length or length 1')
            max_len = max(len(item) for item in param_list_filtered)
            padded_data = [[item[0]] * max_len if len(item) == 1 else item for item in param_list_filtered]
            check_values = np.array(padded_data).T
        else:
            check_values = cartesian_product(*[param_list[i] for i in check_indices]).reshape(-1, len(check_indices))

        checks = []
        for point in self._data_unstructured:
            check = any([np.array_equal(point[check_indices], row) for row in check_values])
            checks.append(check)
            if check:
                data.append(point[0])
        return np.array(data)

    def get_points_structured(self, condition: AnalysisCondition | None = None) -> np.ndarray:
        if self.condition_structured is None:
            return np.array([])
        if condition is None:
            assert self._data_structured is not None
            return self._data_structured

        query_conditions = condition.parameter_list()
        data_conditions = self.condition_structured.parameter_list()
        indices_list = []
        for data, query in zip(data_conditions, query_conditions):
            if len(query) == 0:
                indices_list.append([i for i, _ in enumerate(data)])
            else:
                new_indices = search_sorted_with_nan(data, query)
                new_indices_clean = [i for i in new_indices if not np.isnan(i)]
                indices_list.append(new_indices_clean)

        has_data = all([len(lst) > 0 for lst in indices_list])
        if not has_data:
            return np.array([])
        indices_product = np.array(cartesian_product(*[np.array(i) for i in indices_list]), dtype=int)
        reshape_args = (-1, self._condition_type.num_params())
        assert self._data_structured is not None
        data = self._data_structured[tuple(indices_product.reshape(*reshape_args).T)]
        conditions = self.condition_structured.parameter_grid()[tuple(indices_product.reshape(*reshape_args).T)]
        data = np.hstack((conditions, data))
        return data

    def regularize_unstructured_points(self, condition: AnalysisCondition | None = None):
        condition = copy.deepcopy(condition) if condition is not None else self._condition_type()
        _, _, uniques = self._build_unstructured_interpolator()
        mins = [np.min(u) for u in uniques]
        maxs = [np.max(u) for u in uniques]
        nums = [len(u) for u in uniques]
        input_values = [np.linspace(mn, mx, n) for mn, mx, n in zip(mins, maxs, nums)]
        default_condition = self._condition_type(*input_values)
        condition.fill_with_other(default_condition)
        grid = condition.parameter_grid()
        n = self._condition_type.num_params()
        queries = np.ascontiguousarray(grid.reshape(-1, n), dtype=float)
        result = self.interpolate_general(queries)[:, n:]
        result_grid = result.reshape((*grid.shape[:-1], self._num_outputs))
        self.clear_all()
        self.set_structured_data(result_grid, condition)

    def grad_structured(self, axis: int | str) -> np.ndarray:
        if isinstance(axis, str):
            axis = self._input_var_names.index(axis)
        
        assert isinstance(axis, int)
        if self._gradient_cache['hash'] == hash(self):
            grads = self._gradient_cache['grads']
            assert isinstance(grads, list)
            if grads[axis] is not None:
                return grads[axis]
        else:
            self._gradient_cache['hash'] = hash(self)
            self._gradient_cache['grads'] = [np.array(0.0)] * self.num_inputs

        assert self._data_structured is not None
        grad = ndgrad(self._data_structured, self._condition_structured.parameter_list()[axis], axis)
        assert not isinstance(self._gradient_cache['grads'], int)
        self._gradient_cache['grads'][axis] = grad
        return grad

    def set_structured_data(self, data: np.ndarray, condition: AnalysisCondition):
        assert len(data.shape) == self.num_inputs+1 and data.shape[-1] == self.num_outputs, \
            'data does not match required shape'

        condition = copy.deepcopy(condition)
        condition.set_defaults()

        self._condition_structured = condition
        self._data_structured = np.array(data, dtype=float) if self._allow_interpolation else data
        self._data_all = None

        if self._allow_interpolation:
            axes = tuple([np.array(c, dtype=float) for c in condition.parameter_list()])
            self._data_ndpolator = RegularGridInterpolator(axes, data, bounds_error=False)

    def add_unstructured_data(self, new_points: np.ndarray):
        assert len(new_points.shape) == 2 and new_points.shape[1] == self._num_inputs + self._num_outputs, \
            'data does not match required shape'

        self._data_all = None
        if self._data_unstructured is None:
            self._data_unstructured = new_points
        else:
            self._data_unstructured = np.vstack([self._data_unstructured, new_points])

    def clear_duplicates(self):
        n = self._condition_type.num_params()
        structured = self.get_points_structured()[0][:, n:]
        unstructured = self.get_points_unstructured()[0][:, n:]
        total_data = np.vstack((structured, unstructured))
        _, first_indices = np.unique(total_data, return_index=True, axis=0)
        unique_indices = [f - len(structured) for f in first_indices if f >= len(structured)]
        unstructured_full = self.get_points_unstructured()
        unstructured_polar = unstructured_full[0][unique_indices]
        self.clear_unstructured()
        self.add_unstructured_data(unstructured_polar)

    def clear_structured(self):
        self._condition_structured = None
        self._data_structured = None
        self._data_ndpolator = None
        self._data_all = None

    def clear_unstructured(self):
        self._data_unstructured = None
        self._data_all = None

    def clear_all(self):
        self.clear_unstructured()
        self.clear_structured()

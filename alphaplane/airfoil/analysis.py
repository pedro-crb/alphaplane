from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Optional, Callable

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, to_hex
from itertools import cycle

import pandas as pd
import numpy as np

from alphaplane.airfoil.foil_data import PolarData, BoundaryLayerData
from alphaplane.airfoil.xfoil import xfoil
from alphaplane.geometry.curve2d import Curve2d
from alphaplane.airfoil.foil_condition import AirfoilCondition
from alphaplane.numerical_tools.array_operations import cartesian_product
from alphaplane.airfoil.neuralfoil import run_from_airfoil as run_nf

if TYPE_CHECKING:
    from alphaplane.airfoil.airfoil import Airfoil
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


class AirfoilAnalysis:
    def __init__(self, airfoil: Airfoil) -> None:
        from alphaplane.airfoil.airfoil import Airfoil
        self.airfoil: Airfoil = airfoil
        self.data: PolarData = PolarData()
        self.data_bl: BoundaryLayerData = BoundaryLayerData()

    @property
    def condition(self):
        return self.data.condition

    def run(self, condition: AirfoilCondition, method: str = 'xfoil', **kwargs) -> dict[str, np.ndarray]:
        if method == 'xfoil':
            return self.run_xfoil(condition, **kwargs)
        elif method == 'neuralfoil':
            return self.run_neuralfoil(condition, **kwargs)
        else:
            raise ValueError(f"invalid method {method}. should be either 'xfoil' or 'neuralfoil'")

    def run_neuralfoil(self, condition: AirfoilCondition, model_size: str = 'large',
                       output_unstructured: bool = False, replace_data: bool = True) -> dict[str, np.ndarray]:
        if replace_data:
            self.clear_all_data()

        condition = copy.deepcopy(condition)
        condition.set_defaults()
        result, params = run_nf(self.airfoil, condition, model_size)
        unstructured_data = np.vstack((params.T, result.T))
        if output_unstructured or condition.is_sequence:
            self.data.add_unstructured_data(unstructured_data)
        else:
            if not replace_data:
                new_parameters = condition.parameter_list()
                old_parameters = self.data.condition_structured.parameter_list() \
                    if (self.data.condition_structured is not None) else [[]] * self.data.num_inputs
                new_sets = [set(new) for new in new_parameters]
                old_sets = [set(old) for old in old_parameters]
                full_parameters = [np.sort(list(new.union(old))) for new, old in zip(new_sets, old_sets)]
                condition = AirfoilCondition(*full_parameters) # type: ignore
            result, params = run_nf(self.airfoil, condition, model_size)
            result = result.reshape(*condition.parameter_grid().shape[:-1], self.data.num_outputs)
            self.data.set_structured_data(result, condition)

        return dict(zip(self.data.input_var_names + self.data.output_var_names, unstructured_data))

    def run_xfoil(self, condition: AirfoilCondition, max_iter: int = 100, show_progress: bool = False,
                  timeout_seconds: float = 5, output_unstructured: bool = False, replace_data: bool = True,
                  pre_filter: Callable[[dict], bool] | None = None) -> dict[str, np.ndarray]:
        if replace_data:
            self.clear_all_data()

        if not condition.is_sequence:
            self._run_xfoil_grid(condition, max_iter, show_progress, timeout_seconds, output_unstructured, pre_filter)
        else:
            self._run_xfoil_sequence(condition, max_iter, show_progress, timeout_seconds)

        return dict(zip(self.data.input_var_names + self.data.output_var_names, self.get_raw_points_df()[0].to_numpy().T))

    def _run_xfoil_grid(self, condition: AirfoilCondition, max_iter: int = 100, show_progress: bool = False,
                        timeout_seconds: float = 5, output_unstructured: bool = False,
                        pre_filter: Callable[[dict], bool] | None = None) -> None:

        xfoil.validate_airfoil(self.airfoil)

        new_parameters = condition.parameter_list()
        old_parameters = self.data.condition_structured.parameter_list() \
            if (self.data.condition_structured is not None) and (not output_unstructured) \
            else [[]] * self.data.num_inputs
        new_sets = [set(new) for new in new_parameters]
        old_sets = [set(old) for old in old_parameters]
        full_parameters = [np.sort(list(new.union(old))) for new, old in zip(new_sets, old_sets)]
        condition = AirfoilCondition(*full_parameters) # type: ignore
        condition.set_defaults()

        if not all([new & old == set() for new, old in zip(new_sets, old_sets)]):
            warnings.warn('Found overlap between new conditions and existing conditions.\n'
                          'All overlapping conditions will be recalculated')

        parameter_grid_full = condition.parameter_grid()
        parameter_grid_without_alpha = parameter_grid_full[0, ...]
        parameter_grid_without_values = parameter_grid_without_alpha[..., 0]

        data_grid_interp = np.full((*parameter_grid_full[..., 0].shape, self.data.num_outputs), np.nan)
        data_grid_raw = np.full((*parameter_grid_full[..., 0].shape, self.data.num_outputs), np.nan)
        bl_grid = np.full((*parameter_grid_full[..., 0].shape, self.data_bl.num_outputs), None, dtype='object')
        total_runs = np.size(parameter_grid_without_values)
        prgress_track = 0
        for index, _ in np.ndenumerate(parameter_grid_without_values):
            current_parameters = parameter_grid_without_alpha[index][1:]
            assert condition.alpha is not None
            alphas: np.ndarray = np.array(condition.alpha) # type: ignore
            
            conditions_matrix = np.concatenate(([alphas], np.array([current_parameters] * len(alphas)).T)).T # type: ignore

            if not output_unstructured:
                data_old_polar = [self.data.get_points_structured(AirfoilCondition(*c)) for c in conditions_matrix]
                data_old_bl = [self.data.get_points_structured(AirfoilCondition(*c)) for c in conditions_matrix]
                data_old_polar = [d[self.data.num_inputs:] for d in data_old_polar]
                data_old_bl = [d[self.data_bl.num_inputs:] for d in data_old_bl]
                has_old_data = not all([len(d) == 0 for d in data_old_polar])
            else:
                data_old_polar = None
                data_old_bl = None
                has_old_data = False

            is_old_alpha = [a in old_parameters[0] for a in alphas] if has_old_data else [False] * len(alphas)
            is_new_alpha = [a in new_parameters[0] for a in alphas] if has_old_data else [True] * len(alphas)

            keys = ['alpha', 'reynolds', 'mach', 'deflection', 'hinge_position', 'xtr_top', 'xtr_bottom', 'n_crit']
            valid = [pre_filter(dict(zip(keys, p))) for p in conditions_matrix] \
                if pre_filter is not None else [True] * len(alphas)
            is_new_alpha = np.logical_and(valid, is_new_alpha)

            if show_progress:
                print(f'\rRunning XFOIL on "{self.airfoil.name}". '
                      f'Progress: {100 * prgress_track / total_runs:.2f}%', end='')
            prgress_track += 1

            if not any(valid):
                continue

            run_alphas = alphas[is_new_alpha]
            current_condition = AirfoilCondition(
                run_alphas,
                *current_parameters
            )
            polar_run, bl_data_run = xfoil.run(self.airfoil, current_condition, max_iter, timeout_seconds)

            polar = np.full((alphas.shape[0], polar_run.shape[1]), np.nan)
            bl_data = np.full((alphas.shape[0], bl_data_run.shape[1]), None, dtype='object')

            polar[is_new_alpha] = polar_run
            if any(is_old_alpha):
                polar[is_old_alpha] = np.array([p[0] for i, p in enumerate(data_old_polar) if is_old_alpha[i]]) # type: ignore
                # this isn't a single assignment to avoid conversion to dtype=float
                bl_data_old = np.array([[None] * self.data_bl.num_outputs for i in is_old_alpha if i])
                bl_data_old_list = [p[0] for i, p in enumerate(data_old_bl) if is_old_alpha[i]] # type: ignore
                for i, val in enumerate(bl_data_old):
                    bl_data_old[i] = bl_data_old_list[i]
                bl_data[is_old_alpha] = bl_data_old

            bl_data[is_new_alpha] = bl_data_run
            polar_interp = xfoil.interpolate_polar(alphas, polar, alphas)

            data_grid_raw[valid, *index, :] = polar[valid]
            data_grid_interp[[True] * len(alphas), *index, :] = polar_interp
            bl_grid[valid, *index] = bl_data[valid]

            if output_unstructured:
                polar_unstruct = np.hstack((conditions_matrix, polar))
                bl_unstruct = np.hstack((conditions_matrix, bl_data))
                polar_valid = [all([not np.isnan(p) for p in row]) for row in polar_unstruct]
                bl_valid = [all([p is not None for p in row] for row in bl_unstruct)]
                final_valid = np.logical_and(polar_valid, bl_valid)
                polar_unstruct = polar_unstruct[final_valid]
                bl_unstruct = bl_unstruct[final_valid]
                self.data.add_unstructured_data(polar_unstruct)
                self.data_bl.add_unstructured_data(bl_unstruct)

        if not output_unstructured:
            self.data.set_structured_data(
                data_grid_interp,
                condition,
            )
            self.data_bl.set_structured_data(
                bl_grid,
                condition,
            )

        if show_progress:
            print(f'\rRunning XFOIL on "{self.airfoil.name}". Progress: {100.00}%')

        return None

    def _run_xfoil_sequence(self, condition: AirfoilCondition, max_iter: int = 100, show_progress: bool = False,
                            timeout_seconds: float = 5):

        xfoil.validate_airfoil(self.airfoil)
        condition = copy.deepcopy(condition)
        condition.set_defaults()
        params = condition.parameter_sequence()
        total_runs = len(params)
        progress_track = 0
        for p in params:
            if show_progress:
                print(f'\rRunning XFOIL on "{self.airfoil.name}". '
                      f'Progress: {100 * progress_track / total_runs:.2f}%', end='')
            current_condition = AirfoilCondition(*p)
            polar, bl_data = xfoil.run(self.airfoil, current_condition, max_iter, timeout_seconds)
            valid = all([~np.isnan(i) for i in polar[0]]) and all([i is not None for i in bl_data[0]])
            polar = np.hstack([np.atleast_2d(p), polar])
            bl_data = np.hstack([np.atleast_2d(p), bl_data])
            if valid:
                self.data.add_unstructured_data(polar)
                self.data_bl.add_unstructured_data(bl_data)
            progress_track += 1

        if show_progress:
            print(f'\rRunning XFOIL on "{self.airfoil.name}". Progress: {100.00}%')

    def get_raw_points_df(self, condition: AirfoilCondition | None = None,
                   datasets='all') -> tuple[pd.DataFrame, pd.DataFrame | None]:
        polar_columns = self.data.all_var_names
        polar = self.data.get_points(condition, datasets)
        polar_df = pd.DataFrame(polar, columns=polar_columns)
        bl_columns = self.data_bl.all_var_names
        bl = self.data_bl.get_points(condition, datasets)
        bl_df = None
        if bl.size > 0:
            bl_df = pd.DataFrame(bl, columns=bl_columns)
        return polar_df, bl_df

    def get_interpolated_df(self, condition: AirfoilCondition, method='auto') -> pd.DataFrame:
        condition = copy.deepcopy(condition)

        def build_query(_condition: AirfoilCondition) -> np.ndarray:
            if _condition.is_sequence:
                _query = np.ascontiguousarray(_condition.parameter_sequence())
            else:
                _query = np.ascontiguousarray(_condition.parameter_grid().reshape(-1, self.data.num_inputs))
            return _query

        if (method == 'auto' and self.data.has_unstructured) or method == 'unstructured':
            interpolator = self.data.interpolate_general
            assert self.data.condition_unstructured is not None
            condition.fill_with_other(self.data.condition_unstructured)
        else:
            interpolator = self.data.interpolate_structured
            assert self.data.condition_structured is not None
            condition.fill_with_other(self.data.condition_structured)

        query = build_query(condition)
        data = interpolator(query)
        data_df = pd.DataFrame(data, columns=self.data.all_var_names)
        return data_df

    def get_polars(self, condition: AirfoilCondition | None = None) -> list[pd.DataFrame]:
        condition = copy.deepcopy(condition) if condition is not None else AirfoilCondition()

        if condition.is_sequence:
            alpha = condition.alpha
            condition.alpha = np.array([])  # type: ignore
            sequence = condition.parameter_sequence()
            results = []
            for s in sequence:
                cond = AirfoilCondition(*s)
                cond.alpha = alpha
                cond.is_sequence = False
                new_polar = self.get_polars(cond)
                results = results + new_polar
            return results

        if self.data.has_unstructured:
            assert self.data.condition_unstructured is not None
            condition.fill_with_other(self.data.condition_unstructured)
        else:
            assert self.data.condition_structured is not None
            condition.fill_with_other(self.data.condition_structured)
        params = condition.parameter_list()
        polar_conditions = cartesian_product(*params[1:]).reshape(-1, 7)
        polar_conditions = [AirfoilCondition(params[0], *p) for p in polar_conditions]
        results = [self.get_interpolated_df(c).sort_values('alpha') for c in polar_conditions]
        return results

    def plot_cp_comb(self, condition: AirfoilCondition,
                     rotate_by_alpha: bool = False,
                     label: Optional[str] = None,
                     legend: bool = False, show: bool = True,
                     fig_ax: Optional[tuple[Figure, Axes]] = None) -> tuple[Figure, Axes]:
        assert condition.is_single_point, "Can't plot multiple conditions simultaneously"

        data_bl = self.get_raw_points_df(condition)[1]
        assert data_bl is not None

        ue_vinf = np.array(data_bl['vel_ratio'].values[0], dtype=float)
        cp = 1 - ue_vinf ** 2

        colors = np.array([(0, 'red'), (0.5, 'red'), (0.5, 'blue'), (1.0, 'blue')], dtype=tuple)
        cmap = LinearSegmentedColormap.from_list("discontinuous_blue_red", colors)

        curve = Curve2d(self.airfoil.coords)
        if rotate_by_alpha:
            curve = curve.rotated(data_bl['alpha'].values[0])

        cp = cp[:self.airfoil.num_pts]

        fig, ax = curve.plot_comb(
            data=-cp,
            scale_factor=0.5,
            label=label,
            legend=legend,
            colormap=cmap,
            normalize_cmap=False,
            show=show,
            arrow=True,
            fig_ax=fig_ax
        )
        return fig, ax

    def plot_cp(self, condition: AirfoilCondition,
                colors: str | list[str] = 'default',
                linewidth: float = 1.2,
                label: Optional[str] = None,
                legend: bool = False,
                show: bool = True,
                fig_ax: Optional[tuple[Figure, Axes]] = None
                ) -> tuple[Figure, Axes]:

        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax

        data_bl = self.get_raw_points_df(condition)[1]
        assert data_bl is not None

        if colors == 'default':
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif isinstance(colors, str):
            cmap = plt.get_cmap(colors)
            colors = [to_hex(cmap(i)) for i in np.linspace(0, 1, data_bl.shape[0])]

        for i, (row, color) in enumerate(zip(data_bl.iterrows(), cycle(colors))):
            ue_vinf = np.array(row[1]['vel_ratio'], dtype=float)
            cp = 1 - ue_vinf ** 2
            cp = cp[:self.airfoil.num_pts]

            coords = self.airfoil.coords[:, 0]
            label = f"{self.airfoil.name} alpha={row[1]['alpha']}" if label is None else label
            ax.plot(coords, -cp, color=color, linewidth=linewidth, label=label)

        if show:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True)
            ax.set_xlim([-0.05, 1.05]) # type: ignore
            if legend:
                ax.legend()
            plt.show()
        return fig, ax

    def plot_dstar(self, condition: AirfoilCondition,
                   colors: str | list[str] = 'default',
                   linewidth: float = 1.0,
                   label: Optional[str] = None,
                   legend: bool = False, show: bool = True,
                   rotate_by_alpha: bool = False,
                   fig_ax: Optional[tuple[Figure, Axes]] = None) -> tuple[Figure, Axes]:

        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(14, 8))
        else:
            fig, ax = fig_ax

        data_bl = self.get_raw_points_df(condition)[1]
        assert data_bl is not None

        if colors == 'default':
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif isinstance(colors, str):
            try:
                cmap = plt.get_cmap(colors)
                colors = [to_hex(cmap(i)) for i in np.linspace(0, 1, data_bl.shape[0])]
            except ValueError:
                colors = [colors]

        for i, (row, color) in enumerate(zip(data_bl.iterrows(), cycle(colors))):
            fig_ax = (fig, ax)

            dstar = np.array(row[1]['dstar'], dtype=float)

            curve = Curve2d(self.airfoil.coords)
            if rotate_by_alpha:
                curve = curve.rotated(row[1]['alpha'])

            dstar = dstar[:self.airfoil.num_pts]

            fig, ax = curve.plot_comb(
                data=dstar,
                scale_factor=0.5,
                label=label,
                legend=legend,
                normalize_cmap=False,
                show=False,
                supress_combs=True,
                spline_color=color,
                spline_width=linewidth,
                supress_curve=(not i == data_bl.shape[0] - 1),
                fig_ax=fig_ax
            )

        if show:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True)
            ax.axis('equal')
            ax.set_xlim([-0.01, 1.05]) # type: ignore
            if legend:
                ax.legend()
            plt.show()

        return fig, ax

    def plot_polars(self, condition: AirfoilCondition | None = None, colors: str = 'default', label: str | None = None,
                    legend: bool = False, show: bool = True, fig_axs: tuple[Figure, list[Axes]] | None = None,
                    **kwargs) -> tuple[Figure, list[Axes]]:

        if fig_axs is None:
            fig = plt.figure(figsize=(14, 7))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(2, 4, 3)
            ax3 = fig.add_subplot(2, 4, 4)
            ax4 = fig.add_subplot(2, 4, 7)
            ax5 = fig.add_subplot(2, 4, 8)
            fig_axs = (fig, [ax1, ax2, ax3, ax4, ax5])

        data = self.get_polars(condition)

        if colors == 'default':
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif isinstance(colors, str):
            try:
                cmap = plt.get_cmap(colors)
                colors = [to_hex(cmap(i)) for i in np.linspace(0, 1, len(data))] # type: ignore
            except ValueError:
                colors = [colors] # type: ignore

        for d, color in zip(data, cycle(colors)):
            self._plot_polar(d, color, label, legend, False, fig_axs, **kwargs)

        ax1 = fig_axs[1][0]

        if show:
            for ax in fig_axs[1]:
                ax.grid()
            if legend:
                ax1.legend()
            plt.tight_layout()
            plt.show()

        return fig_axs

    @staticmethod
    def _plot_polar(data: pd.DataFrame, color: str = 'black', label: str | None = None, legend: bool = False,
                    show: bool = True, fig_axs: tuple[Figure, list[Axes]] | None = None,
                    **kwargs) -> tuple[Figure, list[Axes]]:

        alphas = data['alpha'].values
        CL = data['Cl'].values
        CD = data['Cd'].values
        CM = data['Cm'].values
        LD = CL / CD # type: ignore

        if fig_axs is None:
            fig = plt.figure(figsize=(14, 7))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(2, 4, 3)
            ax3 = fig.add_subplot(2, 4, 4)
            ax4 = fig.add_subplot(2, 4, 7)
            ax5 = fig.add_subplot(2, 4, 8)
        else:
            fig = fig_axs[0]
            ax1, ax2, ax3, ax4, ax5 = fig_axs[1]

        ax1.plot(CD, CL, color, label=label, **kwargs) # type: ignore
        ax1.set_title('Cl x Cd')

        grid = fig.add_subplot(1, 2, 2)
        grid.axis('off')

        ax2.plot(alphas, CL, color, label=label, **kwargs) # type: ignore
        ax2.set_title('Cl x alpha')

        ax3.plot(CL, LD, color, label=label, **kwargs) # type: ignore
        ax3.set_title('L/D x Cl')

        ax4.plot(alphas, CD, color, label=label, **kwargs) # type: ignore
        ax4.set_title('Cd x alpha')

        ax5.plot(alphas, CM, color, label=label, **kwargs) # type: ignore
        ax5.set_title('Cm x alpha')

        axes = [ax1, ax2, ax3, ax4, ax5]

        if show:
            for ax in axes:
                ax.grid()
            if legend:
                ax1.legend()
            plt.tight_layout()
            plt.show()

        return fig, axes

    def clear_all_data(self):
        self.data.clear_all()
        self.data_bl.clear_all()

    @staticmethod
    def plot_multiple(foils: list[Airfoil], condition: AirfoilCondition,
                      colors: str | list[str] | tuple[str, list[float] | np.ndarray] = 'default', 
                      legend: bool = True, show: bool = True, fig_axs: tuple[Figure, list[Axes]] | None = None,
                      **kwargs) -> tuple[Figure, list[Axes]]:
        if fig_axs is None:
            fig = plt.figure(figsize=(14, 7))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(2, 4, 3)
            ax3 = fig.add_subplot(2, 4, 4)
            ax4 = fig.add_subplot(2, 4, 7)
            ax5 = fig.add_subplot(2, 4, 8)
            fig_axs = (fig, [ax1, ax2, ax3, ax4, ax5])

        if colors == 'default':
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif isinstance(colors, tuple):
            cmap = plt.get_cmap(colors[0])
            color_var = np.array(colors[1])
            color_var = (color_var - np.min(color_var))/(np.max(color_var) - np.min(color_var))
            colors = [to_hex(cmap(i)) for i in color_var]
        elif isinstance(colors, str):
            try:
                cmap = plt.get_cmap(colors)
                colors = [to_hex(cmap(i)) for i in np.linspace(0, 1, len(foils))]
            except ValueError:
                colors = [colors]

        for foil, color in zip(foils, cycle(colors)):
            foil.analysis.plot_polars(
                condition=condition,
                colors=[color], # type: ignore
                label=foil.name,
                legend=legend,
                show=False,
                fig_axs=fig_axs,
                **kwargs
            )

        ax1 = fig_axs[1][0]

        if show:
            for ax in fig_axs[1]:
                ax.grid(which='both')
                ax.minorticks_on()
                ax.grid(which='minor', linestyle='--', linewidth=0.5, color='#cccccc')
            if legend:
                ax1.legend()
            plt.tight_layout()
            plt.show()

        return fig_axs

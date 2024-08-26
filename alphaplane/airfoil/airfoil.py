from __future__ import annotations

from typing import Callable, Self

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib.colors import to_hex
from collections import deque
from itertools import cycle

import scipy.interpolate as interp
import matplotlib.pyplot as plt
import dill as pickle
import pandas as pd
import numpy as np
import warnings
import bisect
import copy
import os
import re

from alphaplane.numerical_tools.interpolators import ModifiedAkimaSpline, PiecewiseQuinticInterpolator
from alphaplane.numerical_tools.array_operations import monotonic_indices
from alphaplane.numerical_tools.cst import cst, cst_fit_line
from alphaplane.airfoil.foil_condition import AirfoilCondition
from alphaplane.airfoil.analysis import AirfoilAnalysis
from alphaplane.geometry.curve2d import Curve2d


class _AirfoilBase(Curve2d):

    def __init__(self, coords: np.ndarray, name: str = 'Unnamed Airfoil') -> None:
        coords = np.array(coords) + 0.

        self._name: str = name
        self._upper_function: Callable[[float | np.ndarray], float | np.ndarray] | None = None
        self._lower_function: Callable[[float | np.ndarray], float | np.ndarray] | None = None
        self._camber_coords: np.ndarray | None = None

        super().__init__(coords, name)

    def __repr__(self) -> str:
        return f"({self.num_pts}-point {self.__class__.__name__}, name='{self.name}')"

    @property
    def upper_function(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        if self._upper_function is None:
            self._calculate_interpolating_functions()
        return self._upper_function # type: ignore

    @property
    def lower_function(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        if self._lower_function is None:
            self._calculate_interpolating_functions()
        return self._lower_function # type: ignore

    @property
    def upper_coords(self) -> np.ndarray:
        leading_edge_ind = np.argmin(self.coords_x)
        return np.array(self.coords[:leading_edge_ind + 1][::-1]) + 0.

    @property
    def lower_coords(self) -> np.ndarray:
        leading_edge_ind = np.argmin(self.coords_x)
        return np.array(self.coords[leading_edge_ind:]) + 0.

    @property
    def camber_coords(self) -> np.ndarray:
        x = self.upper_coords[:, 0]
        return np.array((self.upper_function(x) + self.lower_function(x)) / 2) + 0.

    @property
    def deflection_angle(self) -> float:
        return 0.0

    @property
    def leading_edge_index(self) -> int:
        return int(np.argmin(self.coords[:, 0]))

    @property
    def name(self) -> str:
        if self._name is None:
            self._name = 'Unnamed Airfoil'
        return self._name

    def to_dat(self, file_path: str) -> None:
        with open(file_path, 'w') as file:
            file.write(f"{self.name}\n")
            for x, y in self.coords:
                file.write(f" {x: 0.8f}    {y: 0.8f}\n")

    def to_csv(self, file_path: str, 
           insert_zeros_column: int | None = None,
           include_header: bool = False,
           ) -> None:
        coords = self.coords
        if insert_zeros_column is not None:
            zeros = np.zeros_like(coords[:, 0])
            coords = np.insert(coords, insert_zeros_column, zeros, axis=1)

        fmt = '%.10f'
        if coords.shape[1] > 1:
            fmt = ','.join([fmt] * coords.shape[1])
        
        header = 'x/c,y/c' if include_header else ''
        np.savetxt(file_path, coords, delimiter=',', header=header, comments='', fmt=fmt)

    def plot(self, color: str = 'black',
             marker: str = '',
             linewidth: float = 1.2,
             label: str | None = None,
             legend: bool = False,
             show: bool = True,
             fig_ax: tuple[Figure, Axes] | None = None
             ) -> tuple[Figure, Axes]:
        fig, ax = self.plot_foil(color=color, marker=marker, linewidth=linewidth, label=label,
                                 legend=legend, plot_camberline=False, show=show, fig_ax=fig_ax)
        return fig, ax

    def plot_foil(self, color: str = 'black',
                  marker: str = '',
                  linewidth: float = 1.2,
                  label: str | None = None,
                  legend: bool = False,
                  plot_camberline: bool = False,
                  show: bool = True,
                  fig_ax: tuple[Figure, Axes] | None = None,
                  **kwargs
                  ) -> tuple[Figure, Axes]:
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax

        coords = self.coords

        label = self.name if label is None else label

        ax.plot(coords[:, 0], coords[:, 1], color=color, marker=marker, linestyle='-',
                linewidth=linewidth, label=label, **kwargs)

        if plot_camberline:
            camber = self.camber_coords
            ax.plot(camber[:, 0], camber[:, 1], color=color, marker=marker,
                    linestyle='--', linewidth=linewidth, **kwargs)

        if show:
            ax.axis('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True)
            ax.set_title('Airfoil Geometry')
            if legend:
                ax.legend()
            plt.show()

        return fig, ax

    @staticmethod
    def plot_multiple(foils: list,
                      colors: str | list[str] | tuple[str, list[float | np.ndarray]] = 'default',
                      marker: str = '',
                      labels: list[str] | None = None,
                      legend: bool = True,
                      plot_camberline: bool = False,
                      show: bool = True,
                      fig_ax: tuple[Figure, Axes] | None = None,
                      **kwargs
                      ) -> tuple[Figure, Axes]:
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax

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

        for i, (foil, color) in enumerate(zip(foils, cycle(colors))):

            if labels is None:
                label = foil.name
            else:
                label = labels[i]

            foil.plot_foil(color=color, marker=marker, label=label, legend=False,
                           plot_camberline=plot_camberline, show=False, fig_ax=(fig, ax), **kwargs)

        if show:
            ax.axis('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True)
            ax.set_title('Airfoil Geometry')
            if legend:
                ax.legend()
            plt.show()

        return fig, ax

    def plot_curvature(self, show: bool = True,
                       fig_ax: tuple[Figure, Axes] | None = None
                       ) -> tuple[Figure, Axes]:
        fig, ax = self.plot_comb(self.curvature, scale_factor=0.01, colormap='jet', show=show, fig_ax=fig_ax)
        return fig, ax

    def plot_panel_angles(self, show: bool = True,
                          fig_ax: tuple[Figure, Axes] | None = None
                          ) -> tuple[Figure, Axes]:
        fig, ax = self.plot_colorline(self.edge_angles, colormap='jet', show=show, fig_ax=fig_ax)
        return fig, ax

    def _calculate_interpolating_functions(self) -> None:
        num_pts = len(self.coords)
        coords = self.coords
        x = coords[:, 0]
        y = coords[:, 1]
        x = x
        y = y
        x_min = np.min(x)
        x_max = np.max(x)
        x_range = x_max - x_min

        t = self.arclength / np.max(self.arclength)
        valid = monotonic_indices(t)
        t = t[valid]
        x = x[valid]
        y = y[valid]
        spline_x = interp.CubicSpline(t, x)
        spline_y = interp.CubicSpline(t, y)

        # supersample the spline
        supersample = max(4, 2000 // num_pts)
        factors = np.linspace(0, 1, supersample + 1)[:-1].reshape(-1, 1)
        t_samples = (1 - factors) * t[:-1].reshape(1, -1) + factors * np.roll(t, -1)[:-1].reshape(1, -1)
        t_samples = np.append(t_samples.flatten(order='F'), t[-1])

        refined_x = spline_x(t_samples)
        refined_y = spline_y(t_samples)
        refined_leading_edge_index = np.argmin(refined_x)
        refined_x_min = np.min(refined_x)
        refined_x_max = np.max(refined_x)
        refined_x_range = refined_x_max - refined_x_min
        refined_x = (refined_x - refined_x_min) * (x_range / refined_x_range) + x_min
        refined_coords = np.array([refined_x, refined_y]).T
        refined_upper = refined_coords[:refined_leading_edge_index + 1][::-1]
        refined_lower = refined_coords[refined_leading_edge_index:]
        refined_upper = refined_upper[monotonic_indices(refined_upper[:, 0])]
        refined_lower = refined_lower[monotonic_indices(refined_lower[:, 0])]

        self._upper_function = ModifiedAkimaSpline(
            refined_upper[:, 0],
            refined_upper[:, 1],
        )
        self._lower_function = ModifiedAkimaSpline(
            refined_lower[:, 0],
            refined_lower[:, 1],
        )


class Airfoil(_AirfoilBase):

    AirfoilAnalysis = AirfoilAnalysis

    def __init__(self, coords: np.ndarray, name: str = 'Unnamed Airfoil') -> None:
        from alphaplane.airfoil.analysis import AirfoilAnalysis
        coords = np.array(coords) + 0.
        super().__init__(coords, name)
        self.analysis: AirfoilAnalysis = AirfoilAnalysis(self)

        self._deflected_cache = deque(maxlen=10)
        self._neuralfoil_parameters = None

    @property
    def camber_max(self) -> float:
        return float(self.camber_coords[:, 1][np.argmax(np.abs(self.camber_coords[:, 1]))])

    @property
    def thickness_max(self) -> float:
        return np.max(self.thickness_function(self.upper_coords[:, 0]))

    @property
    def camber_function(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        return lambda x: (self.upper_function(x) + self.lower_function(x)) / 2

    @property
    def thickness_function(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        return lambda x: self.upper_function(x) - self.lower_function(x)

    @property
    def camber_coords(self) -> np.ndarray:
        return np.array([self.upper_coords[:, 0], self.camber_function(self.upper_coords[:, 0])]).T

    @property
    def camber_pos(self) -> float:
        return float(self.upper_coords[np.argmax(self.camber_coords[:, 1]), 0])

    @property
    def thickness_pos(self) -> float:
        thickness = self.thickness_function(self.upper_coords[:, 0])
        return float(self.upper_coords[np.argmax(thickness), 0])

    @property
    def trail_edge_gap(self) -> float:
        return float(self.thickness_function(max(float(self.coords[0, 0]), float(self.coords[-1, 0]))))

    def cst_parameters(self, camber_num_parameters: int, thickness_num_parameters: int) -> tuple[np.ndarray, ...]:
        foil = copy.deepcopy(self)
        foil.set_TE_gap(0.0)
        camber_x = foil.camber_coords[:, 0]
        camber_y = foil.camber_coords[:, 1]
        thickness_x = camber_x
        thickness_y = foil.thickness_function(thickness_x)
        _x = (thickness_x-np.min(thickness_x))/(np.max(thickness_x) - np.min(thickness_x))
        thickness_y_delta = thickness_y - _x*foil.trail_edge_gap
        camber_parameters = cst_fit_line(camber_x, camber_y, camber_num_parameters, 1.0, 1.0)
        thickness_parameters = cst_fit_line(thickness_x, thickness_y_delta, thickness_num_parameters, 0.5, 1.0)
        return camber_parameters, thickness_parameters

    def neuralfoil_parameters(self) -> dict[str, np.ndarray | float]:
        if self._neuralfoil_parameters is not None:
            return self._neuralfoil_parameters
        foil = copy.deepcopy(self)
        foil.repair_geometry()
        if foil.num_pts < 150:
            foil.repanel(150)
        foil.set_TE_gap(0.0)
        upper_weights = cst_fit_line(foil.upper_coords[:, 0], foil.upper_coords[:, 1], 8, 0.5, 1.0)
        lower_weights = cst_fit_line(foil.lower_coords[:, 0], foil.lower_coords[:, 1], 8, 0.5, 1.0)
        self._neuralfoil_parameters = {
            "lower_weights": lower_weights,
            "upper_weights": upper_weights,
            "TE_thickness": self.trail_edge_gap,
            "leading_edge_weight": 0.0,
        }
        return self._neuralfoil_parameters

    def deflected(self, deflection_angle: float, hinge_position: float = 0.7) -> FlappedAirfoil:

        if self._deflected_cache is not None:
            for angle, hinge, result in self._deflected_cache:
                if angle == deflection_angle and hinge == hinge_position:
                    return result

        if deflection_angle == 0.0:
            return FlappedAirfoil(self.coords,
                                  self.camber_coords,
                                  deflection_angle,
                                  hinge_position,
                                  self)

        # points at x_hinge
        hinge_upper = np.array([hinge_position, self.upper_function(hinge_position)])
        hinge_lower = np.array([hinge_position, self.lower_function(hinge_position)])
        hinge_mid = np.array([hinge_position, self.camber_function(hinge_position)])

        # split foil into 3 segments
        flap_upper = self.upper_coords[self.upper_coords[:, 0] > hinge_position]
        fixed_points = self.coords[self.coords[:, 0] < hinge_position]
        flap_lower = self.lower_coords[self.lower_coords[:, 0] > hinge_position]
        # split camber into 2 segments
        camber_fixed = self.camber_coords[self.camber_coords[:, 0] < hinge_position]
        camber_flap = self.camber_coords[self.camber_coords[:, 0] > hinge_position]

        # add hinge points to curves
        flap_upper = np.vstack([hinge_upper, flap_upper])
        fixed_points = np.vstack([hinge_upper, fixed_points, hinge_lower])
        flap_lower = np.vstack([hinge_lower, flap_lower])
        camber_fixed = np.vstack([camber_fixed, hinge_mid])
        camber_flap = np.vstack([hinge_mid, camber_flap])

        # rotate coordinates
        hinge_point = hinge_lower if deflection_angle > 0 else hinge_upper
        theta = np.deg2rad(deflection_angle)
        R = np.array([[np.cos(-theta), -np.sin(-theta)],
                      [np.sin(-theta), np.cos(-theta)]])
        flap_upper = np.dot(flap_upper - hinge_point, R.T) + hinge_point
        flap_lower = np.dot(flap_lower - hinge_point, R.T) + hinge_point
        camber_flap = np.dot(camber_flap - hinge_point, R.T) + hinge_point

        if deflection_angle > 0:
            coords_deflected = np.vstack([
                Curve2d.from_curve_join(flap_upper[::-1], fixed_points, 3).coords, flap_lower[1:]
            ])
        else:
            coords_deflected = np.vstack([
                flap_upper[1:][::-1], Curve2d.from_curve_join(fixed_points, flap_lower, 3).coords
            ])
        camber_deflected = Curve2d.from_curve_join(camber_fixed, camber_flap, 3).coords

        # merge points to repair geometry
        coords_curve = Curve2d(coords_deflected)
        coords_curve.merge_points(
            tolerance=np.min(self.edge_sizes) / 10,
            preserved_indices=[self.leading_edge_index])
        coords_deflected = coords_curve.coords
        ##
        camber_curve = Curve2d(camber_deflected)
        camber_curve.merge_points(
            tolerance=np.min(self.edge_sizes) / 10)
        camber_deflected = camber_curve.coords

        flapped_airfoil = FlappedAirfoil(coords_deflected, camber_deflected,
                                         deflection_angle, hinge_position, self)
        self._deflected_cache.append((deflection_angle, hinge_position, flapped_airfoil))
        return flapped_airfoil

    def repair_geometry(self) -> None:
        self.derotate()
        if self.coords[0, 1] != self.coords[-1, 1]:
            self.align_TE_coords()
            self.derotate()
        self.normalize()
        self.merge_points()

    def repanel(self, num_pts: int, bunching_strength: tuple[float, float] = (1.0, 0.0),
                repair: bool = False) -> None:

        if repair:
            self.repair_geometry()

        num_pts = num_pts if (num_pts % 2) else num_pts + 1
        num_x = 1 + num_pts // 2
        x = self.make_x_distribution(num_x, bunching_strength)

        # use the distribution within the range of each side of the airfoil
        min_x_upper = np.min(self.upper_coords[:, 0])
        max_x_upper = np.max(self.upper_coords[:, 0])
        min_x_lower = np.min(self.lower_coords[:, 0])
        max_x_lower = np.max(self.lower_coords[:, 0])
        x_upper = x * (max_x_upper - min_x_upper) + min_x_upper
        x_lower = x * (max_x_lower - min_x_lower) + min_x_lower
        upper = np.array(self.upper_function(x_upper))
        lower = np.array(self.lower_function(x_lower))
        x_final = np.concatenate([x_upper[::-1], x_lower[1:]])
        y_final = np.concatenate([upper[::-1], lower[1:]])
        coords = np.array([x_final, y_final]).T
        self.set_coords(coords)

        if repair:
            self.repair_geometry()

    def scale_camber_and_thickness(
            self, newcamber: float | None = None, newcamberpos: float | None = None,
            newthickness: float | None = None, newthicknesspos: float | None = None
    ) -> None:

        camber_function = self.camber_function # type: ignore
        thickness_function = self.thickness_function # type: ignore

        x_min = np.min(self.coords[:, 0])
        x_max = np.max(self.coords[:, 0])

        if newcamberpos is not None:
            camber_pos_function = (
                interp.interp1d(
                    [x_min, newcamberpos, x_max], [x_min, self.camber_pos, x_max], bounds_error=False
                )
            )

            def camber_function(x):
                return self.camber_function(camber_pos_function(x))

        if newthicknesspos is not None:
            thickness_pos_function = (
                interp.interp1d(
                    [x_min, newthicknesspos, x_max], [x_min, self.thickness_pos, x_max], bounds_error=False
                )
            )

            def thickness_function(x):
                return self.thickness_function(thickness_pos_function(x))

        if newcamber is not None:
            if self.camber_max == 0.0:
                raise ValueError(f'Can\'t scale camber of uncambered airfoil.\n'
                                 f'Use {self.__class__.__name__}.add_camber instead')
            camber_scale = newcamber / self.camber_max
            if self.camber_max < 1e-4 and camber_scale > 5:
                warnings.warn(f'Attempting to scale camber by {100 * camber_scale:.2f}%.\n'
                              f'To add camber to uncambered airfoil, use {self.__class__.__name__}.add_camber')
        else:
            camber_scale = 1.0

        thickness_scale = newthickness / self.thickness_max if newthickness is not None else 1.0

        x_upper = self.upper_coords[:, 0]
        x_lower = self.lower_coords[:, 0]
        upper = np.array((camber_scale * camber_function(self.upper_coords[:, 0]) +
                 thickness_scale * thickness_function(self.upper_coords[:, 0]) / 2))

        lower = np.array((camber_scale * camber_function(self.lower_coords[:, 0]) -
                 thickness_scale * thickness_function(self.lower_coords[:, 0]) / 2))

        x_final = np.concatenate([x_upper[::-1], x_lower[1:]])
        y_final = np.concatenate([upper[::-1], lower[1:]])
        coords = np.array([x_final, y_final]).T

        self.set_coords(coords)

    def add_camber(self, added_camber: float, camber_pos: float | None = None) -> None:
        minx = np.min(self.coords[:, 0])
        maxx = np.max(self.coords[:, 0])

        if camber_pos is None and self.camber_max < 1e-4:
            raise ValueError('To add camber to uncambered airfoil, a camber_pos must be specified')

        if camber_pos is None:
            camber_pos = self.camber_pos

        # function to control added camber positioning
        f = interp.interp1d([minx, camber_pos, maxx], [0.0, 1.0, 0.0])

        def add_coords(_coords, camber):
            add = camber * (1 - (1 - f(_coords[:, 0])) ** 2)
            y_new = _coords[:, 1] + add
            return np.vstack([_coords[:, 0], y_new]).T

        upper_new = add_coords(self.upper_coords, added_camber)
        lower_new = add_coords(self.lower_coords, added_camber)
        coords = np.concatenate([upper_new[::-1], lower_new[1:]])

        self.set_coords(coords)

    def scale_LE_radius(self, scale_factor: float, blend_start: float = 0.2,
                        blend_end: float = 0.5) -> None:
        x_interpolation = np.array([self.x_min, blend_start, blend_end, self.x_max])
        y_interpolation = np.array([np.sqrt(scale_factor), np.sqrt(scale_factor), 1.0, 1.0])
        derivatives = np.array([0.0, 0.0, 0.0, 0.0])
        thickness_scale_function = (
            PiecewiseQuinticInterpolator(x_interpolation, y_interpolation, derivatives, derivatives)
        )

        def offset_funtion(x):
            return self.thickness_function(x) * (thickness_scale_function(x) - 1) / 2

        upper_new_y = self.upper_coords[:, 1] + offset_funtion(self.upper_coords[:, 0])
        lower_new_y = self.lower_coords[:, 1] - offset_funtion(self.lower_coords[:, 0])

        upper_new = np.vstack((self.upper_coords[:, 0], upper_new_y)).T
        lower_new = np.vstack((self.lower_coords[:, 0], lower_new_y)).T

        coords = np.concatenate([upper_new[::-1], lower_new[1:]])

        self.set_coords(coords)

    def set_TE_gap(self, new_TE_gap: float,
                   gap_blend_start: float = 0.2,
                   fix_thickness: bool = True) -> None:
        gap_change = new_TE_gap - self.trail_edge_gap
        min_x = np.min(self.coords[:, 0])
        max_x = np.max(self.coords[:, 0])
        gap_change_function = (
            interp.interp1d([min_x, gap_blend_start, max_x], [0.0, 0.0, gap_change], bounds_error=False)
        )

        # doesnt allow for thicknesses smaller than 80% of the gap
        x_upper = self.upper_coords[:, 0]
        x_lower = self.lower_coords[:, 0]
        gap_offset_upper = (1 / 2) * np.maximum(
            gap_change_function(x_upper),
            (0.8 * new_TE_gap - self.thickness_function(x_upper)) * np.where(x_upper > self.thickness_pos, 1, 0)
        ) if fix_thickness else 0.5 * gap_change_function(x_upper)
        gap_offset_lower = (1 / 2) * np.maximum(
            gap_change_function(x_lower),
            (0.8 * new_TE_gap - self.thickness_function(x_lower)) * np.where(x_lower > self.thickness_pos, 1, 0)
        ) if fix_thickness else 0.5 * gap_change_function(x_lower)

        upper = self.upper_coords[:, 1] + gap_offset_upper
        lower = self.lower_coords[:, 1] - gap_offset_lower
        x_final = np.concatenate([x_upper[::-1], x_lower[1:]])
        y_final = np.concatenate([upper[::-1], lower[1:]])
        coords = np.array([x_final, y_final]).T

        self.set_coords(coords)

    def derotate(self) -> None:
        coords = self.coords
        trailing_edge = np.array([coords[0], coords[-1]])
        trailing_edge_point = (trailing_edge[0] + trailing_edge[1]) / 2
        distances = np.linalg.norm(coords - trailing_edge_point, axis=1)

        # Set the leading edge to be the farthest point from the trailing edge
        leading_edge_point = coords[np.argmax(distances)]

        coords_translated = coords - leading_edge_point
        trailing_edge_point_translated = trailing_edge_point - leading_edge_point
        theta = np.arctan2(trailing_edge_point_translated[1], trailing_edge_point_translated[0])
        R = np.array([[np.cos(-theta), -np.sin(-theta)],
                      [np.sin(-theta), np.cos(-theta)]])
        coords = np.dot(coords_translated + leading_edge_point, R.T)
        self.set_coords(coords)

    def normalize(self) -> None:
        coords = self.coords
        leading_edge_point = coords[np.argmin(coords[:, 0])]
        coords -= leading_edge_point
        leading_edge_point -= leading_edge_point
        # rescale coords to bring the rightmost point to x-coordinate 1.0
        trailing_edge_x = np.max(coords[:, 0])
        coords /= trailing_edge_x

        self.set_coords(coords)

    def align_TE_coords(self) -> None:
        """Fixes the trailing edge by scaling the first or last panel to match both trailing edge points x-position"""

        if self.coords[0, 0] < self.coords[-1, 0]:
            working_indices = [1, 0]
            TE_x = self.coords[-1, 0]
        elif self.coords[0, 0] > self.coords[-1, 0]:
            working_indices = [-2, -1]
            TE_x = self.coords[0, 0]
        else:
            return
        coords = self.coords

        delta = coords[working_indices[1]] - coords[working_indices[0]]
        delta_x_target = TE_x - coords[working_indices[0], 0]
        delta *= delta_x_target / delta[0]
        coords[working_indices[1]] = coords[working_indices[0]] + delta
        coords[working_indices[1], 0] = TE_x

        self.set_coords(coords)

    def merge_points(self, tolerance: float = 1e-4,
                     preserved_indices: list[int] | None = None) -> None:
        if preserved_indices is None:
            preserved_indices = []
        bisect.insort(preserved_indices, self.leading_edge_index)

        super().merge_points(
            tolerance=tolerance, preserved_indices=preserved_indices)

    def invert(self) -> None:
        coords = self.coords
        coords[:, 1] = -coords[:, 1]
        coords = coords[::-1]
        self.set_coords(coords)

    def smooth_airfoil(self, window_size: int | None = None,
                       blend_factor: float = 0.4, repetitions: int = 5,
                       repair: bool = True):

        if window_size is None:
            window_size = min(self.num_pts // 50, 3)

        if repair:
            self.repair_geometry()

        for _ in range(repetitions):
            coords = self.coords

            if window_size % 2 == 0:
                window_size += 1

            pad_width = window_size // 2
            padded_coords = np.pad(coords, ((pad_width, pad_width), (0, 0)), mode='edge')

            # rolling average
            kernel = np.ones(window_size) / window_size
            smoothed_coords_x = np.convolve(padded_coords[:, 0], kernel, mode='valid')
            smoothed_coords_y = np.convolve(padded_coords[:, 1], kernel, mode='valid')
            smoothed_coords = np.vstack((smoothed_coords_x, smoothed_coords_y)).T
            coords = (1 - blend_factor) * coords + blend_factor * smoothed_coords

            # morph new foil to make it match with old foil
            working_foil = self.from_coords(coords)
            working_LE = working_foil.coords[working_foil.leading_edge_index]
            working_TE = (working_foil.coords[0] + working_foil.coords[-1]) / 2
            original_LE = self.coords[self.leading_edge_index]
            original_TE = (self.coords[0] + self.coords[-1]) / 2
            working_foil_curve = working_foil.transformed(
                source=np.array([working_LE, working_TE]),
                destination=np.array([original_LE, original_TE])
            )
            working_foil = self.from_coords(working_foil_curve.coords)
            working_foil.set_TE_gap(self.trail_edge_gap)
            if self.camber_max > 5e-4:
                working_foil.scale_camber_and_thickness(
                    newcamber=self.camber_max, newthickness=self.thickness_max,
                    newcamberpos=self.camber_pos, newthicknesspos=self.thickness_pos,
                )
            else:
                working_foil.scale_camber_and_thickness(
                    newthickness=self.thickness_max,
                    newthicknesspos=self.thickness_pos,
                )

            self.set_coords(working_foil.coords)

        if repair:
            self.repair_geometry()

    def to_pickle(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def from_coords(cls, coords: np.ndarray,
                    name: str = 'Unnamed Airfoil') -> Self:
        return cls(coords, name)

    @classmethod
    def from_naca(cls, naca_digits: str, num_pts: int = 201, TE_gap: float = 0.0,
                  bunching_strength: tuple[float, float] = (1.0, 0.0), name: str | None = None) -> Self:

        naca_digits = ''.join(re.findall(r'\d+', str(naca_digits)))

        num_pts = num_pts if (num_pts % 2) else num_pts + 1
        num_x = 1 + num_pts // 2
        x = cls.make_x_distribution(num_x, bunching_strength)

        a0 = +0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = +0.2843
        a4 = -0.1036 + TE_gap / (100 * 2 * 5 * (int(naca_digits[-2:]) / 100.0))
        yt = 5 * (int(naca_digits[-2:]) / 100.0) * (a0 * np.sqrt(x) + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4)

        if len(naca_digits) == 4:
            m = float(naca_digits[0]) / 100.0
            p = float(naca_digits[1]) / 10.0
            xc1 = x[x <= p]
            xc2 = x[x > p]
            yc1 = m / p ** 2 * xc1 * (2 * p - xc1) if p != 0 else np.zeros_like(xc1)
            yc2 = m / (1 - p) ** 2 * (1 - 2 * p + xc2) * (1 - xc2) if p != 0 else np.zeros_like(xc2)
            dyc1_dx = m / p ** 2 * (2 * p - 2 * xc1) if p != 0 else - 2 * xc1
            dyc2_dx = m / (1 - p) ** 2 * (2 * p - 2 * xc2)
            zc = np.concatenate([yc1, yc2]) if p != 0 else np.zeros_like(x)

        elif len(naca_digits) == 5:
            naca1 = int(naca_digits[0])
            naca23 = int(naca_digits[1:3])
            cld = naca1 * (3.0 / 2.0) / 10.0
            p = 0.5 * naca23 / 100.0
            P = [0.05, 0.1, 0.15, 0.2, 0.25]
            M = [0.0580, 0.1260, 0.2025, 0.2900, 0.3910]
            K = [361.4, 51.64, 15.957, 6.643, 3.230]
            m = interp.interp1d(P, M, kind='cubic', bounds_error=False)(p)
            k1 = interp.interp1d(M, K, kind='cubic', bounds_error=False)(m)
            xc1 = x[x <= p]
            xc2 = x[x > p]
            yc1 = k1 / 6.0 * (xc1 ** 3 - 3 * m * xc1 ** 2 + m ** 2 * (3 - m) * xc1)
            yc2 = k1 / 6.0 * m ** 3 * (1 - xc2)
            dyc1_dx = cld / 0.3 * (1.0 / 6.0) * k1 * (3 * xc1 ** 2 - 6 * m * xc1 + m ** 2 * (3 - m))
            dyc2_dx = np.full(len(xc2), cld / 0.3 * -(1.0 / 6.0) * k1 * m ** 3)
            zc = cld / 0.3 * np.concatenate([yc1, yc2])

        else:
            raise ValueError('naca must be 4 or 5 digits')

        dyc_dx = np.concatenate([dyc1_dx, dyc2_dx])
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = zc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = zc - yt * np.cos(theta)

        X = np.concatenate([xu[::-1], xl[1:]])
        Y = np.concatenate([yu[::-1], yl[1:]])
        coords = np.array([X, Y]).T

        if name is None:
            name = f'NACA {naca_digits}'

        foil = cls(coords, name)
        foil.set_TE_gap(1e-30)
        foil.repair_geometry()
        return foil

    @classmethod
    def from_database(cls, airfoil_name: str, name: str | None = None) -> Self:

        if not airfoil_name.endswith('.dat'):
            airfoil_name += '.dat'

        if name is None:
            name = airfoil_name.rsplit('.', 1)[0]

        script_dir = os.path.dirname(os.path.abspath(__file__))
        airfoil_path = os.path.join(script_dir, 'selig_database', airfoil_name)

        if not os.path.isfile(airfoil_path):
            raise Exception(f"Airfoil '{airfoil_name}' not found in database")

        airfoil = cls.from_dat(airfoil_path, name)
        return airfoil

    @classmethod
    def from_dat(cls, file_path: str, name: str | None = None) -> Self:
        DEFAULT_NAME = 'DAT Airfoil'

        with open(file_path, 'r') as file:
            lines = file.readlines()

        first_line_parts = lines[0].strip().split()
        if all(part.replace('.', '', 1).isdigit() or
               part.lstrip('-').replace('.', '', 1).isdigit()
               for part in first_line_parts):
            name = DEFAULT_NAME if name is None else name
            data_start_index = 0
        else:
            name = lines[0].strip() if name is None else name
            data_start_index = 1

        x_values, y_values = [], []
        for line in lines[data_start_index:]:
            if line.strip():
                try:
                    x, y = map(float, line.split())
                    x_values.append(x)
                    y_values.append(y)
                except ValueError:
                    continue

        coords = np.array([x_values, y_values]).T

        # treating lednicer format dat files
        if all(coords[0] >= 2 * np.max(coords[1:, 0])):
            coords = coords[1:]
        x = coords[:, 0]
        dx = np.diff(x)
        if np.max(np.abs(dx)) > 0.5 * (np.max(x) - np.min(x)):
            ind = np.argmax(np.abs(dx))
            if ind + 1 < len(x) and np.abs(x[0] - x[ind + 1]) < np.abs(x[ind] - x[ind + 1]):
                upper = coords[0:ind + 1][::-1]
                lower = coords[ind + 2:]
                coords = np.concatenate([upper, lower])

        return cls(coords, name)

    @classmethod
    def from_csv(cls, file_path: str, xcol: int = 0, ycol: int = 1, name: str | None = None) -> Self:
        df = pd.read_csv(file_path)
        coords = df.iloc[:, [xcol, ycol]].to_numpy()
        return cls(coords, name) # type: ignore

    @classmethod
    def from_cst(cls, camber_parameters: np.ndarray | list[float],
                 thickness_parameters: np.ndarray | list[float],
                 num_pts: int = 201, TE_gap: float = 0.0,
                 bunching_strength: tuple[float, float] = (1.0, 0.0),
                 name: str = 'CST Airfoil') -> Self:
        num_pts = num_pts if (num_pts % 2) else num_pts + 1
        num_x = 1 + num_pts // 2

        x = cls.make_x_distribution(num_x, bunching_strength)
        camber = cst(x, np.array(camber_parameters), n1=1.0, n2=1.0)
        thickness = cst(x, np.array(thickness_parameters), n1=0.5, n2=1.0)
        thickness += TE_gap * (x - np.min(x))/(np.max(x) - np.min(x))
        upper = camber + thickness / 2
        lower = camber - thickness / 2

        x_final = np.concatenate([x[::-1], x[1:]])
        y_final = np.concatenate([upper[::-1], lower[1:]])

        coords = np.array([x_final, y_final]).T

        return cls(coords, name)

    @classmethod
    def from_camber_and_thickness(
            cls, x: np.ndarray | list[float],
            camber: np.ndarray, thickness: np.ndarray,
            name: str = 'Unnamed Airfoil') -> Self:
        upper = camber + thickness / 2
        lower = camber - thickness / 2

        x_final = np.concatenate([x[::-1], x[1:]])
        y_final = np.concatenate([upper[::-1], lower[1:]])

        coords = np.array([x_final, y_final]).T

        return cls(coords, name)

    @classmethod
    def from_upper_and_lower(cls, upper: np.ndarray, lower: np.ndarray,
                             name: str = 'Unnamed Airfoil') -> Self:
        coords = np.concatenate([upper[::-1], lower[1:]])
        return cls(coords, name)

    @classmethod
    def from_foil_interpolation(cls, foils: list[Airfoil],
                                weights: list[float] | np.ndarray | None = None,
                                num_pts: int | None = None,
                                bunching_strength: tuple[float, float] = (1.0, 0.0),
                                interpolate_polars: bool = False,
                                name: str = 'Interpolated Airfoil') -> Self:
        if weights is None:
            weights = np.ones(len(foils))

        if num_pts is None:
            num_pts = max([foil.num_pts for foil in foils])

        weights = np.array(weights)
        weights = weights / np.sum(weights)
        coords = None

        foil_copies = []
        for foil, weight in zip(foils, weights): # type: ignore
            new_foil = copy.deepcopy(foil)
            new_foil.repanel(num_pts, bunching_strength)
            new_coords = new_foil.coords
            coords = weight * new_coords if coords is None else coords + weight * new_coords
            foil_copies.append(new_foil)

        final_foil = cls(coords, name) # type: ignore

        if interpolate_polars:
            interp_condition = AirfoilCondition.from_intersection([foil.analysis.condition for foil in foils])

            grid = interp_condition.parameter_grid()
            queries = np.ascontiguousarray(grid.reshape(-1, 8), dtype=float)
            polar_grid = None
            for foil, w in zip(foil_copies, weights): # type: ignore
                if foil.analysis.data.has_unstructured:
                    foil.analysis.data.regularize_unstructured_points()
                result = foil.analysis.data.interpolate_structured(queries)[:, 8:]
                result = result.reshape((*grid.shape[:-1], 8))
                polar_grid = result * w if polar_grid is None else polar_grid + result * w
            final_foil.analysis.data.set_structured_data(polar_grid, interp_condition) # type: ignore
        return final_foil

    @classmethod
    def from_ellipse(cls, thickness: float = 0.2, num_pts: int = 201,
                     bunching_strength: tuple[float, float] = (1.0, 0.0), name: str = 'Ellipse Airfoil') -> Self:
        num_pts = num_pts if (num_pts % 2) else num_pts + 1
        num_x = 1 + num_pts // 2
        x = cls.make_x_distribution(num_x, bunching_strength)

        thickness_array = thickness * np.sqrt(1 - (2 * x - 1) ** 2)
        camber_array = np.zeros_like(thickness_array)

        return cls.from_camber_and_thickness(x, camber_array, thickness_array, name=name)

    @classmethod
    def from_pickle(cls, file_path: str) -> Self:
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    @classmethod
    def from_neuralfoil(cls, neuralfoil_params: dict, num_pts: int = 201, name: str = 'Neuralfoil Airfoil'):
        num_pts = num_pts if (num_pts % 2) else num_pts + 1
        num_x = 1 + num_pts // 2
        x = cls.make_x_distribution(num_x)
        upper = cst(x, neuralfoil_params['upper_weights'], 0.5, 1.0)
        lower = cst(x, neuralfoil_params['lower_weights'], 0.5, 1.0)
        coords_x = np.concatenate([x[::-1], x[1:]])
        coords_y = np.concatenate([upper[::-1], lower[1:]])
        coords = np.vstack((coords_x, coords_y)).T
        foil = cls(coords, name)
        foil.set_TE_gap(neuralfoil_params['TE_thickness'], fix_thickness=False)
        return foil

    def set_coords(self, new_coords: np.ndarray) -> None:
        analysis = self.analysis
        super().set_coords(new_coords)
        self.analysis = analysis
        self._deflected_cache = deque(maxlen=10)

    @staticmethod
    def make_x_distribution(num_x: int, bunching_strength: tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
        a0 = 1 + np.sqrt(bunching_strength[0])
        a1 = 1 + np.sqrt(bunching_strength[1])
        x = np.linspace(0, 1, num_x)
        x = x ** a0 / (x ** a0 + (1 - x) ** a1)
        return x

    @staticmethod
    def get_database_list() -> list[str]:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        database_path = os.path.join(script_dir, 'selig_database')
        database_list = []
        for filename in os.listdir(database_path):
            database_list.append(filename.replace('.dat', ''))
        return database_list


class FlappedAirfoil(_AirfoilBase):
    def __init__(self,
                 coords_deflected: np.ndarray,
                 camber_deflected: np.ndarray,
                 deflection_angle: float,
                 hinge_position: float,
                 primitive_airfoil: Airfoil):
        super().__init__(coords_deflected, primitive_airfoil.name)

        # properties for the deflected geometry
        self._coords: np.ndarray = coords_deflected
        self._camber_coords: np.ndarray = camber_deflected
        self._deflection_angle: float = deflection_angle
        self._hinge_position: float = hinge_position
        self._primitive_airfoil: Airfoil = primitive_airfoil

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        coords_equal = np.array_equal(self.coords, other.coords)
        hinge_equal = self.hinge_position == other.hinge_position
        angle_equal = self.deflection_angle == other.deflection_angle
        return coords_equal and hinge_equal and angle_equal

    @property
    def deflection_angle(self) -> float:
        return self._deflection_angle

    @property
    def hinge_position(self) -> float:
        return self._hinge_position

    @property
    def camber_coords(self) -> np.ndarray:
        return self._camber_coords

    @property
    def primitive_airfoil(self) -> Airfoil:
        return copy.deepcopy(self._primitive_airfoil)

    @property
    def name(self) -> str:
        sign_str = '+' if self.deflection_angle >= 0.0 else '-'
        self._name = f'{self.primitive_airfoil.name}{sign_str}{abs(int(self.deflection_angle)):02d}'
        return self._name

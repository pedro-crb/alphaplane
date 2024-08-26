from __future__ import annotations
from typing import Callable, Self, TYPE_CHECKING

if TYPE_CHECKING:
    from alphaplane.propeller.analysis import PropellerAnalysis

from scipy.interpolate import interp1d, PchipInterpolator, CubicHermiteSpline
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import matplotlib.pyplot as plt
import dill as pickle
import pandas as pd
import numpy as np
import warnings
import json
import copy
import csv
import os

from alphaplane.numerical_tools.interpolators import ModifiedAkimaSpline
from alphaplane.airfoil.airfoil import Airfoil


class Propeller:
    def __init__(self,
                 n_blades: int,
                 stations: np.ndarray,
                 chords: np.ndarray,
                 twists: np.ndarray,
                 airfoils: list[tuple[float, Airfoil]] | None = None,
                 offsets_x: np.ndarray | None = None,
                 offsets_z: np.ndarray | None = None,
                 thickness: np.ndarray | None = None,
                 weight: float | None = None,
                 name: str | None = 'Unnamed Propeller'
                 ) -> None:
        offsets_x = np.zeros_like(stations) if offsets_x is None else offsets_x
        offsets_z = np.zeros_like(stations) if offsets_z is None else offsets_z

        # basic properties
        self._name: str = name if name is not None else 'Unnamed Propeller'
        self._n_blades: int = n_blades
        self._stations: np.ndarray = stations
        self._chords: np.ndarray = chords
        self._twists: np.ndarray = twists
        if airfoils is None:
            self._airfoils = [(0.0, Airfoil.from_naca('4412', 151))]
        else:
            self._airfoils: list[tuple[float, Airfoil]] = airfoils
        self._airfoils.sort(key=lambda section: section[0])
        self._offsets_x: np.ndarray = offsets_x
        self._offsets_z: np.ndarray = offsets_z
        self._thickness: np.ndarray | None = thickness

        # hub properties
        self._hub_dimaeter: float | None = None
        self._hub_thickness: float | None = None
        self._hub_thickness_scale: float | None = None
        self._hub_twist: float | None = None
        self._hub_n_stations: int | None = None
        self.reset_hub()

        # mass properties
        self._density_points: list[tuple[float, float]] | None = None
        self._weight: float | None = weight

        # stored geometry data
        self._points3d: np.ndarray | None = None
        self._points3d_with_transition: np.ndarray | None = None
        self._stations_with_transition: np.ndarray | None = None

        # analysis
        from alphaplane.propeller.analysis import PropellerAnalysis
        self.analysis: PropellerAnalysis = PropellerAnalysis(self)

    def __repr__(self) -> str:
        return f"{self.diameter_inches:.2f} x {self.pitch75_inches:.2f} Propeller, name='{self.name}'"

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_blades(self) -> int:
        return self._n_blades

    @property
    def stations(self) -> np.ndarray:
        return np.array(self._stations, copy=True)

    @property
    def chords(self) -> np.ndarray:
        return np.array(self._chords, copy=True)

    @property
    def twists(self) -> np.ndarray:
        return np.array(self._twists, copy=True)

    @property
    def offsets_x(self) -> np.ndarray:
        return np.array(self._offsets_x, copy=True)

    @property
    def offsets_z(self) -> np.ndarray:
        return np.array(self._offsets_z, copy=True)

    @property
    def thickness(self) -> np.ndarray:
        if self._thickness is None:
            if len(self._airfoils) == 1:
                airfoil = self._airfoils[0][1]
                thickness = airfoil.thickness_max
                thickness = np.ones_like(self.stations) * thickness
                return thickness

            stations, airfoils = zip(*self.airfoils)
            thickness = [foil.thickness_max for foil in airfoils]
            fill_value = (thickness[0], thickness[-1])
            thickness_function = interp1d(
                stations, thickness,
                fill_value=fill_value,  # type: ignore
                bounds_error=False)
            return thickness_function(self.stations)

        return np.array(self._thickness, copy=True)

    @property
    def airfoils(self) -> list[tuple[float, Airfoil]]:
        return self._airfoils

    @property
    def diameter(self) -> float:
        return 2 * np.max(self.stations)

    @property
    def radius(self) -> float:
        return max(self.stations)

    @property
    def chords_function(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        return ModifiedAkimaSpline(self.stations, self.chords)

    @property
    def twists_function(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        return interp1d(self.stations, self.twists)

    @property
    def offsets_x_function(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        return ModifiedAkimaSpline(self.stations, self.offsets_x)

    @property
    def offsets_z_function(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        return ModifiedAkimaSpline(self.stations, self.offsets_z)

    @property
    def thickness_function(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        fill_value = (self.thickness[0], self.thickness[-1])
        thickness = interp1d(self.stations, self.thickness,
                             fill_value=fill_value, bounds_error=False)  # type: ignore
        return thickness

    @property
    def pitch75(self) -> float:
        r = 0.75 * self.diameter / 2
        beta = self.twists_function(r)
        return float(2 * np.pi * r * np.tan(np.radians(beta)))

    @property
    def diameter_inches(self) -> float:
        return self.diameter / 0.0254

    @property
    def pitch75_inches(self) -> float:
        return self.pitch75 / 0.0254

    @property
    def hub_diameter(self) -> float | None:
        return self._hub_dimaeter

    @property
    def hub_thickness(self) -> float | None:
        return self._hub_thickness

    @property
    def points3d(self):
        if self._points3d is None:
            self._make_points3d()
        return copy.deepcopy(self._points3d)

    @property
    def points3d_with_transition(self):
        if self._points3d_with_transition is None:
            if any(i is None for i in [self.hub_diameter, self.hub_thickness]):
                raise NameError(f'Hub geometry is not defined.\n'
                                f'Use {self.__class__.__name__}.set_root to define it')
            self.set_root(self.hub_diameter, self.hub_thickness)
        return copy.deepcopy(self._points3d_with_transition)

    @property
    def stations_with_transition(self):
        if self._stations_with_transition is None:
            if any(i is None for i in [self.hub_diameter, self.hub_thickness]):
                raise NameError(f'Hub geometry is not defined.\n'
                                f'Use {self.__class__.__name__}.set_root to define it')
            self.set_root(self.hub_diameter, self.hub_thickness)
        return copy.deepcopy(self._stations_with_transition)

    @property
    def density_function(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        if self._density_points is None:
            raise AssertionError(
                'A density distribution is not defined for this propeller')

        stations = [point[0] for point in self._density_points]
        densities = [point[1] for point in self._density_points]
        fill_values = (densities[0], densities[-1])
        density = interp1d(
            stations, densities, fill_value=fill_values, bounds_error=False)  # type: ignore
        return density

    @property
    def weight(self) -> float | None:
        return self._weight

    def rename(self, name: str) -> None:
        self._name = name

    def set_n_blades(self, n_blades: int) -> None:
        self._n_blades = n_blades

    def set_blade(self, *,
                  stations: np.ndarray | None = None,
                  chords: np.ndarray | None = None,
                  twists: np.ndarray | None = None,
                  offsets_x: np.ndarray | None = None,
                  offsets_z: np.ndarray | None = None,
                  thickness: np.ndarray | None = None) -> None:
        stations = self.stations if stations is None else stations
        parameters = {'stations': stations, 'chords': chords, 'twists': twists,
                      'offsets_x': offsets_x, 'offsets_z': offsets_z, 'thickness': thickness}
        provided_parameters = {k: v for k,
                               v in parameters.items() if v is not None}

        if any(len(v) != len(stations) for v in provided_parameters.values()):
            raise ValueError('The length of the new arrays must match')
        for attr, value in provided_parameters.items():
            setattr(self, f'_{attr}', value)
        self._clear_geometry_data()

    def insert_blade_sections(self, *,
                              stations: np.ndarray,
                              chords: np.ndarray,
                              twists: np.ndarray,
                              offsets_x: np.ndarray | None = None,
                              offsets_z: np.ndarray | None = None,
                              thickness: np.ndarray | None = None
                              ) -> None:

        if np.min(stations) < np.min(self.stations):
            self.reset_hub()

        offsets_x = np.zeros_like(stations) if offsets_x is None else offsets_x
        offsets_z = np.zeros_like(stations) if offsets_z is None else offsets_z

        if thickness is None:
            thickness = np.array(self.thickness_function(stations))

        parameters = {
            'chords': chords, 'twists': twists,
            'offsets_x': offsets_x, 'offsets_z': offsets_z, 'thickness': thickness
        }

        if any(len(v) != len(stations) for v in parameters.values()):
            raise ValueError(
                'The length of the new arrays must match the length of the new stations')

        self._stations = np.concatenate((self._stations, stations))
        sorted_indices = np.argsort(self._stations)
        self._stations = self._stations[sorted_indices]
        for attr, new_values in parameters.items():
            current_values = getattr(self, f'_{attr}')
            new_values = np.concatenate((current_values, new_values))
            sorted_array = new_values[sorted_indices]
            setattr(self, f'_{attr}', sorted_array)

        self._clear_geometry_data()
        self._clear_analyzed_foils()

    def set_airfoils(self, airfoils: Airfoil | list[tuple[float, Airfoil]],
                     replace_existing=True):

        if isinstance(airfoils, Airfoil):
            if not replace_existing:
                raise AssertionError('if a single airfoil is passed without corresponding stations, '
                                     'replace_existing must be True')
            airfoils = [(0.0, airfoils)]

        if replace_existing:
            self._airfoils = airfoils
        else:
            self._airfoils = self._airfoils + airfoils
            self._airfoils.sort(key=lambda foil_section: foil_section[0])
        self._clear_geometry_data()
        self._clear_analyzed_foils()

    def set_root(self, hub_diameter: float | None = None,
                 hub_thickness: float | None = None,
                 hub_thickness_scale: float | None = None,
                 hub_twist: float | None = None,
                 hub_n_stations: int | None = None) -> None:
        hub_diameter = self._hub_dimaeter if hub_diameter is None else hub_diameter
        hub_thickness = self._hub_thickness if hub_thickness is None else hub_thickness
        hub_thickness_scale = self._hub_thickness_scale if hub_thickness_scale is None else hub_thickness_scale
        hub_twist = self._hub_twist if hub_twist is None else hub_twist
        hub_n_stations = self._hub_n_stations if hub_n_stations is None else hub_n_stations

        if any(i is None for i in [hub_diameter, hub_thickness]):
            raise NameError(f'Hub geometry is not defined.\n'
                            f'Use {self.__class__.__name__}.set_root and pass in at least \n'
                            f'hub_diameter and hub_thickness to define it')
        assert hub_diameter is not None
        assert hub_n_stations is not None
        assert hub_twist is not None

        self._hub_dimaeter = hub_diameter
        self._hub_thickness = hub_thickness
        self._hub_thickness_scale = hub_thickness_scale
        self._hub_twist = hub_twist
        self._hub_n_stations = hub_n_stations
        r_transition = np.min(self.stations)
        r_inner = min(r_transition, hub_diameter / 2)
        new_stations = np.linspace(
            0.0, r_transition, hub_n_stations, endpoint=False)

        chords_new = interp1d(
            [0.0, r_inner, r_transition, self.stations[1], self.stations[2]],
            [
                0.90 * hub_diameter,
                0.85 * hub_diameter,
                self.chords[0],
                self.chords[1],
                self.chords[2]
            ],
            kind='cubic'
        )(new_stations)

        twists_new = ModifiedAkimaSpline(
            [0.0, r_transition, self.stations[1], self.stations[2]],
            [
                hub_twist,
                self.twists[0],
                self.twists[1],
                self.twists[2]
            ],
        )(new_stations)

        root_offset_x = -0.2 * hub_diameter
        offsets_x_new = interp1d(
            [0.0, r_inner, r_transition, self.stations[1], self.stations[2]],
            [root_offset_x, root_offset_x, self.offsets_x[0],
                self.offsets_x[1], self.offsets_x[2]],
            kind='cubic'
        )(new_stations)

        root_offset_z = 0.0
        offsets_z_new = PchipInterpolator(
            np.array([0.0, r_inner, r_transition,
                     self.stations[1], self.stations[2]]),
            np.array([root_offset_z, root_offset_z, self.offsets_z[0],
                     self.offsets_z[1], self.offsets_z[2]]),
        )(new_stations)

        t_old = self.thickness[0]
        t_new = hub_thickness_scale * t_old
        thickness_new = PchipInterpolator(
            np.array([0.0, r_inner, (r_inner + r_transition)/2, r_transition,
                     self.stations[1], self.stations[2]]),
            np.array([t_new, 0.6 * t_new + 0.4 * t_old, 0.15*t_new + 0.85*t_old, self.thickness[0],
                      self.thickness[1], self.thickness[2]]),
        )(new_stations)

        dummy_prop = copy.deepcopy(self)
        dummy_prop.insert_blade_sections(stations=new_stations,
                                         chords=chords_new, twists=twists_new,
                                         offsets_x=offsets_x_new, offsets_z=offsets_z_new,
                                         thickness=thickness_new)

        root_foil = copy.deepcopy(dummy_prop.airfoils[0][1])
        new_airfoil = Airfoil.from_ellipse(0.2, root_foil.num_pts)
        new_airfoil.set_TE_gap(0.02)
        dummy_prop.set_airfoils(airfoils=[(r_transition / 3, new_airfoil), (r_transition / 1.5, root_foil)],
                                replace_existing=False)

        self._points3d_with_transition = dummy_prop.points3d
        self._stations_with_transition = dummy_prop.stations

    def show(self, include_root: bool = False) -> None:
        import pyvista as pv
        points = self.points3d_with_transition if include_root else self.points3d
        assert points is not None

        mesh_points = np.vstack(list(points))
        n_points_airfoil = points[0].shape[0]

        faces = []
        for i in range(len(points) - 1):
            for j in range(n_points_airfoil - 1):
                p1 = i * n_points_airfoil + j
                p2 = (i + 1) * n_points_airfoil + j
                p3 = (i + 1) * n_points_airfoil + (j + 1)
                p4 = i * n_points_airfoil + (j + 1)
                faces.append([4, p1, p4, p3, p2])
            if i < len(points) - 2:
                p1 = i * n_points_airfoil
                p2 = (i + 1) * n_points_airfoil
                p3 = (i + 1) * n_points_airfoil + (n_points_airfoil - 1)
                p4 = i * n_points_airfoil + (n_points_airfoil - 1)
                faces.append([4, p1, p2, p3, p4])

        # Define cap faces correctly
        first_cap_faces = [n_points_airfoil] + list(range(n_points_airfoil))
        last_cap_start = len(mesh_points) - n_points_airfoil
        last_cap_faces = [n_points_airfoil] + \
            list(range(last_cap_start, last_cap_start + n_points_airfoil))

        # Correctly adding cap faces to the faces list
        cap_faces = np.concatenate((
            np.array(first_cap_faces, dtype=np.int64).reshape(-1, ),
            np.array(last_cap_faces, dtype=np.int64).reshape(-1, )
        ), axis=0)

        faces = np.hstack([np.hstack(faces), cap_faces])

        single_blade_mesh = pv.PolyData(mesh_points, faces).clean()
        assert single_blade_mesh is not None
        all_meshes = [single_blade_mesh]
        angle_per_blade = 360 / self.n_blades
        for n in range(1, self.n_blades):
            blade_mesh = single_blade_mesh.copy().rotate_y(angle_per_blade * n, inplace=True)
            if blade_mesh.n_points == 0:
                raise ValueError(
                    f"Rotated blade mesh {n} is empty after triangulation.")
            all_meshes.append(blade_mesh.clean())

        if include_root:
            assert self.hub_diameter is not None and self.hub_thickness is not None
            hub = pv.Cylinder(center=(0, 0, 0), direction=(0, 1, 0),
                              radius=self.hub_diameter / 2, height=self.hub_thickness)
            all_meshes.append(hub)

        mesh = pv.MultiBlock(all_meshes).combine()
        assert mesh is not None
        mesh.plot(show_grid=True)

    def get_interpolated_airfoils(self, stations: list[float] | np.ndarray) -> list[Airfoil]:
        airfoil_stations = np.array([foil_section[0]
                                    for foil_section in self.airfoils])
        airfoils = [copy.deepcopy(foil_section[1])
                    for foil_section in self.airfoils]
        calculated_airfoils = []
        stations = np.array(stations).reshape((-1,))

        if not airfoils:
            raise ValueError('Airfoils list is empty')

        if len(airfoils) == 1:
            for station in stations:
                airfoil = airfoils[0]
                if self.thickness is not None:
                    airfoil.scale_camber_and_thickness(
                        newthickness=float(self.thickness_function(station)))
                calculated_airfoils.append(copy.deepcopy(airfoil))
            return calculated_airfoils

        if np.any(stations < np.min(self.stations)) or np.any(stations > np.max(self.stations)):
            raise ValueError(
                "Some stations are out of the blade's defined range.")

        for station in stations:
            if station <= airfoil_stations[0]:
                airfoil = copy.deepcopy(airfoils[0])
                if self.thickness is not None:
                    airfoil.scale_camber_and_thickness(
                        newthickness=float(self.thickness_function(station)))
            elif station >= airfoil_stations[-1]:
                airfoil = copy.deepcopy(airfoils[-1])
                if self.thickness is not None:
                    airfoil.scale_camber_and_thickness(
                        newthickness=float(self.thickness_function(station)))
            else:
                i = np.searchsorted(airfoil_stations, station)
                f = (station - airfoil_stations[i - 1]) / \
                    (airfoil_stations[i] - airfoil_stations[i - 1])
                airfoil1 = copy.deepcopy(airfoils[i - 1])
                airfoil2 = copy.deepcopy(airfoils[i])
                if self.thickness is not None:
                    airfoil1.scale_camber_and_thickness(
                        newthickness=self.thickness_function(airfoil_stations[i - 1]))
                    airfoil2.scale_camber_and_thickness(
                        newthickness=self.thickness_function(airfoil_stations[i]))
                airfoil = Airfoil.from_foil_interpolation(
                    foils=[airfoil1, airfoil2], weights=[1 - f, f])

            calculated_airfoils.append(airfoil)

        return calculated_airfoils

    def crosssec_area(self, stations: list[float] | np.ndarray | None = None) -> np.ndarray:
        if stations is None:
            stations = self.stations
        foils = self.get_interpolated_airfoils(stations)
        chords = self.chords_function(np.array(stations))
        crosssec_areas = np.array([foil.area for foil in foils]) * chords ** 2
        return crosssec_areas

    def set_offset(self, offsets_x: np.ndarray | float | None = None,
                   offsets_z: np.ndarray | float | None = None,
                   local_orientation: bool = False,
                   local_scale: bool = False,
                   replace: bool = True) -> None:

        offsets_x = np.zeros_like(
            self.stations) if offsets_x is None else np.atleast_1d(offsets_x)
        offsets_z = np.zeros_like(
            self.stations) if offsets_z is None else np.atleast_1d(offsets_z)

        offsets_x = np.ones_like(
            self.stations) * offsets_x[0] if len(offsets_x) == 1 else offsets_x
        offsets_z = np.ones_like(
            self.stations) * offsets_z[0] if len(offsets_z) == 1 else offsets_z

        assert len(np.array(offsets_x)) == len(np.array(offsets_z)) == len(self.stations), \
            'new offsets and propeller stations must have the same length'

        angles_rad = np.radians(
            self.twists) if local_orientation else np.zeros_like(self.twists)
        scale = self.chords if local_scale else np.ones_like(self.chords)
        offsets_x_final = scale * \
            (offsets_x * np.cos(angles_rad) + offsets_z * np.sin(angles_rad))
        offsets_z_final = scale * \
            (offsets_z * np.cos(angles_rad) - offsets_x * np.sin(angles_rad))

        if not replace:
            offsets_x_final += self.offsets_x
            offsets_z_final += self.offsets_z

        self.set_blade(offsets_x=offsets_x_final, offsets_z=offsets_z_final)

    def add_tip_sweep(self, offset: float, blend_distance: float) -> None:
        sweeps_add_function = PchipInterpolator(
            np.array([0.0, self.radius - blend_distance, self.radius]),
            np.array([0.0, 0.0, offset])
        )

        offsets_new = self.offsets_x + sweeps_add_function(self.stations)
        self.set_blade(offsets_x=offsets_new)

    def repanel_airfoils(self, num_pts: int, bunching_strength: tuple[float, float] = (1.0, 0.0),
                         repair: bool = True) -> None:
        _, airfoils_list = zip(*self._airfoils)
        for airfoil in airfoils_list:
            airfoil.repanel(
                num_pts, bunching_strength=bunching_strength, repair=repair)
        self._clear_geometry_data()
        self._clear_analyzed_foils()

    def refine_stations(self, num_stations: int, bunching_strength: tuple[float, float] = (1.0, 2.0)) -> None:
        new_stations = self._make_stations_distribution(
            self.diameter, 2 *
            float(self.stations[0]), num_stations, bunching_strength
        )
        new_chords = self.chords_function(new_stations)
        new_twists = self.twists_function(new_stations)
        new_offsets_x = self.offsets_x_function(new_stations)
        new_offsets_z = self.offsets_z_function(new_stations)
        new_thickness = self.thickness_function(new_stations)

        self._stations = new_stations
        self._chords = np.array(new_chords)
        self._twists = np.array(new_twists)
        self._offsets_x = np.array(new_offsets_x)
        self._offsets_z = np.array(new_offsets_z)
        self._thickness = np.array(new_thickness)

        self._clear_geometry_data()

    def scale(self, new_diameter: float, chords_scale_factor: float | None = None):
        scale_factor = new_diameter / self.diameter
        chords_scale_factor = scale_factor if chords_scale_factor is None else chords_scale_factor
        new_stations = self.stations * scale_factor
        new_chords = self.chords * chords_scale_factor
        new_twists = self.twists
        new_offsets_x = self.offsets_x * chords_scale_factor
        new_offsets_z = self.offsets_z * scale_factor
        new_thickness = self.thickness
        new_airfoils = [(scale_factor * r, foil)
                        for (r, foil) in self.airfoils]
        self.set_blade(
            stations=new_stations,
            chords=new_chords,
            twists=new_twists,
            offsets_x=new_offsets_x,
            offsets_z=new_offsets_z,
            thickness=new_thickness
        )
        self.set_airfoils(new_airfoils)

    def set_airfoils_TE(self, TE_gap: float) -> None:
        _, airfoils_list = zip(*self._airfoils)
        for airfoil in airfoils_list:
            airfoil.set_TE_gap(TE_gap)
        self._clear_geometry_data()
        self._clear_analyzed_foils()

    def set_density(self, density_points: list[tuple[float, float]]) -> None:
        self._density_points = density_points

    def estimate_weights(self) -> list[tuple[float, float]]:
        stations = self.stations
        stations_root = np.linspace(0.0, stations[0], 4, endpoint=False)
        areas = self.crosssec_area(stations)
        areas_root = np.linspace(2 * areas[0], areas[0], 4, endpoint=False)
        stations = np.concatenate([stations_root, stations])
        areas = np.concatenate([areas_root, areas])
        densities = np.array(self.density_function(stations))
        dr = np.diff(stations)
        mass = 2 * dr * areas[:-1] * densities[:-1]
        weights = [(r, m) for r, m in zip(stations, mass)]
        return weights

    def calculate_weight(self) -> None:
        weights = self.estimate_weights()
        m = [w[1] for w in weights]
        self._weight = np.sum(m)

    def plot_at_stations(self, variable: list[float] | np.ndarray, label: str = 'variable',
                         fig_ax: tuple[Figure, Axes] | None = None, show: bool = True) -> tuple[Figure, Axes]:
        if fig_ax is not None:
            fig, ax = fig_ax
        else:
            fig, ax = plt.subplots()
        ax.plot(self.stations, variable, '-')
        if show:
            ax.set_title('distribution along radius')
            ax.set_xlabel('r')
            ax.set_ylabel(label)
            ax.grid(True)
            plt.show()
        return fig, ax

    def reset_hub(self) -> None:
        self._hub_dimaeter = None
        self._hub_thickness = None
        self._hub_thickness_scale = 2
        self._hub_twist = 0.0
        self._hub_n_stations = 10

    def _make_points3d(self, pts_per_airfoil: int = 151) -> None:
        dummy_prop = copy.deepcopy(self)
        dummy_prop.repanel_airfoils(pts_per_airfoil)
        full_airfoils_list = dummy_prop.get_interpolated_airfoils(
            dummy_prop.stations)

        foils3d = []
        num_pts_airfoil = None
        for foil, station, chord, twist, off_x, off_z in (
                zip(full_airfoils_list, dummy_prop.stations,
                    dummy_prop.chords, dummy_prop.twists, dummy_prop.offsets_x, dummy_prop.offsets_z)):
            coords2d = foil.translated(
                [-0.25, 0]).scaled(chord).rotated(twist).translated([off_x, off_z]).coords
            coords3d = np.hstack([coords2d, np.zeros((coords2d.shape[0], 1))])
            coords3d += np.array([0.0, 0.0, station])
            foils3d.append(coords3d)
            if num_pts_airfoil is not None and num_pts_airfoil != foil.num_pts:
                warnings.warn('\n Airfoils do not have the same number of points! \n'
                              'Consider running .repanel_airfoils() before generating mesh')
            num_pts_airfoil = foil.num_pts

        self._points3d = np.array(foils3d)

    def _clear_geometry_data(self) -> None:
        self._points3d = None
        self._weight = None
        self._points3d_with_transition = None
        self._stations_with_transition = None

    def _clear_analyzed_foils(self):
        self.analysis.clear_analyzed_airfoils()

    def to_csv(self, output_path: str, include_root: bool = False,
               include_TE_gap: float = 0.01) -> None:
        dummy_prop = copy.deepcopy(self)
        if include_TE_gap > 0:
            dummy_prop.set_airfoils_TE(include_TE_gap)

        points3d = self.points3d_with_transition if include_root else self.points3d
        stations = self.stations_with_transition if include_root else self.stations
        assert stations is not None and points3d is not None
        with open(output_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(stations)
            rows = []
            n = points3d[0].shape[0]
            for i in range(n):
                row = []
                for points in points3d:
                    points = np.array(points)
                    row.extend(points[i, 0:2])
                rows.append(row)
            csvwriter.writerows(rows)

    def to_pickle_full(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    def to_pickle(self, file_path):
        geometry_dict = {
            'name': self._name,
            'n_blades': self._n_blades,
            'stations': self._stations,
            'chords': self._chords,
            'twists': self._twists,
            'airfoils_stations': [x[0] for x in self._airfoils],
            'airfoils_airfoils': [x[1].coords for x in self._airfoils],
            'offsets_x': self._offsets_x,
            'offsets_z': self._offsets_z,
            'thickness': self._thickness,
            'hub_dimaeter': self._hub_dimaeter,
            'hub_thickness': self._hub_thickness,
            'hub_thickness_scale': self._hub_thickness_scale,
            'hub_twist': self._hub_twist,
            'hub_n_stations': self._hub_n_stations,
            'density_points': self._density_points,
            'weight': self._weight,
        }
        with open(file_path, 'wb') as file:
            pickle.dump(geometry_dict, file)

    def to_flowunsteady(self, working_directory):
        # Create necessary directories
        rotors_dir = os.path.join(working_directory, 'rotors')
        airfoils_dir = os.path.join(working_directory, 'airfoils')
        os.makedirs(rotors_dir, exist_ok=True)
        os.makedirs(airfoils_dir, exist_ok=True)

        # Generate foil stations and interpolated airfoils
        foil_stations = np.linspace(self.stations[0], self.stations[-1], 8)
        airfoils = self.get_interpolated_airfoils(foil_stations)
        foil_stations = np.concatenate([[0.0], foil_stations])
        airfoils = [airfoils[0]] + airfoils

        # Save airfoil files
        for i, airfoil in enumerate(airfoils):
            filename = os.path.join(airfoils_dir, f'{self.name}_foil{i+1}.csv')
            airfoil.to_csv(filename, include_header=True)

        open(os.path.join(airfoils_dir, f'{self.name}_foil-aero.csv'), 'w').close()

        # Prepare data for rotor files
        r = self.radius
        c = self.chords/r
        twist_rad = np.radians(self.twists)
        data = {
            'chorddist': {'r/R': self.stations/r, 'c/R': c},
            'pitchdist': {'r/R': self.stations/r, 'beta': self.twists},
            'sweepdist': {'r/R': self.stations/r, 'sweep': -self.offsets_x/r + 0.25*c*np.cos(twist_rad)},
            'heightdist': {'r/R': self.stations/r, 'height': self.offsets_z/r},
            'airfoil_files': {
                'r/R': foil_stations/r,
                'Contour file': [f'{self.name}_foil{i+1}.csv' for i in range(len(foil_stations))],
                'Aero file': [f'{self.name}_foil-aero.csv']*len(foil_stations)
            }
        }

        # Save rotor files
        for key, values in data.items():
            filename = os.path.join(rotors_dir, f'{self.name}_{key}.csv')
            pd.DataFrame(values).to_csv(filename, index=False)

        # Save propeller main file
        prop_main = pd.DataFrame({
            'property': ['Rtip', 'Rhub', 'B', 'blade'],
            'file': [self.radius, self.stations[0], self.n_blades, f'{self.name}_blade.csv'],
            'description': ['(m) Radius of blade tip', '(m) Radius of hub', 'Number of blades', 'Blade file']
        })
        prop_main.to_csv(os.path.join(
            rotors_dir, f'{self.name}.csv'), index=False)

        # Save blade file
        blade_file = pd.DataFrame({
            'property': ['chorddist', 'pitchdist', 'sweepdist', 'heightdist', 'airfoil_files', 'spl_k', 'spl_s'],
            'file': [
                f'{self.name}_chorddist.csv',
                f'{self.name}_pitchdist.csv',
                f'{self.name}_sweepdist.csv',
                f'{self.name}_heightdist.csv',
                f'{self.name}_airfoil_files.csv',
                4, 5.0e-7],
            'description': ['Chord distribution', 'Pitch distribution', 'LE sweep distribution', 'LE height distribution', 'Airfoil distribution', 'Spline order', 'Spline smoothing']
        })
        blade_file.to_csv(os.path.join(
            rotors_dir, f'{self.name}_blade.csv'), index=False)

    @classmethod
    def from_APC_file(cls, file_path, name=None, convert_inches=True, repanel_args=(151,)) -> Self:
        length_conversion = 0.0254 if convert_inches else 1.0

        airfoil_def = []
        stations = []
        chords = []
        twists = []
        offsets_x_local = []
        thickness = []
        n_blades = []
        weight = []
        _name = []

        def parse_airfoil_line(line):
            parts = line.split(':')
            if len(parts) <= 1:
                return

            station_name_split = parts[1].strip().split(',')
            if len(station_name_split) < 2:
                return

            station = float(station_name_split[0].strip()) * length_conversion
            airfoil_name = station_name_split[1].split()[0].strip()

            if airfoil_name == 'APC12':
                airfoil_name = 'NACA2412'

            if airfoil_name == 'CLARK-Y':
                airfoil_name = 'clarky'

            airfoil = Airfoil.from_database(airfoil_name)

            if repanel_args:
                airfoil.repanel(*repanel_args)

            airfoil_def.append((station, airfoil))

        def parse_prop_data_line(line):
            parts = line.split()
            c = float(parts[1]) * length_conversion
            stations.append(float(parts[0]) * length_conversion)
            chords.append(c)
            twists.append(float(parts[7]))
            offsets_x_local.append(
                0.25 * c - float(parts[5]) * length_conversion)
            thickness.append(float(parts[6]))

        def parse_n_blades_line(line):
            parts = line.split(':')
            n_blades.append(int(parts[1].strip().split()[0]))

        def parse_weight_line(line):
            parts = line.split('=')
            weight.append(float(parts[1].strip()))

        def parse_line(line):
            if name is None:
                _name.append(line.split(' ', 1)[0])

            if line.startswith((' AIRFOIL', 'AIRFOIL')):
                parse_airfoil_line(line)
            elif line.strip() and all(
                    part.replace('.', '', 1).isdigit() or part.startswith('-')
                    for part in line.split()):
                parse_prop_data_line(line)
            elif line.startswith((' BLADES', 'BLADES')):
                parse_n_blades_line(line)
            elif line.startswith((' TOTAL WEIGHT (Kg)', 'TOTAL WEIGHT (Kg)')):
                parse_weight_line(line)

        with open(file_path, 'r') as file:
            for lin in file:
                parse_line(lin)

        name = _name[0] if name is None else name

        new_prop = cls(
            n_blades=n_blades[0],
            stations=np.array(stations),
            chords=np.array(chords),
            twists=np.array(twists),
            airfoils=airfoil_def,
            thickness=np.array(thickness),
            weight=weight[0],
            name=name
        )

        new_prop.set_offset(
            offsets_x=np.array(offsets_x_local),
            local_orientation=False
        )

        return new_prop

    @classmethod
    def from_APC_database(cls, propeller_name: str, load_polars: bool = True, name: str | None = None) -> Self:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        propeller_filename = propeller_name.replace('.', '')
        geometry_path = os.path.join(
            script_dir, 'apc_database', 'geometry', f'{propeller_filename}-PERF.PE0')
        if not os.path.isfile(geometry_path):
            raise Exception(
                f"Propeller '{propeller_name}' not found in database")
        prop = cls.from_APC_file(geometry_path, name)

        if load_polars:
            polar_path = os.path.join(
                script_dir, 'apc_database', 'performance', f'PER3_{propeller_filename}.dat')
            if not os.path.isfile(polar_path):
                raise Exception(
                    f"Performance data for '{propeller_name}' not found in database")
            prop.analysis.polars_from_APC(polar_path)

        return prop

    @classmethod
    def from_qprop(cls, file_path, airfoil: Airfoil) -> Self:
        stations = []
        chords = []
        twists = []
        airfoils = [(0.0, airfoil)]
        name = 'Unnamed Propeller'
        n_blades = 0

        def parse_line(line, count):
            nonlocal name, n_blades
            if not line.strip() or line.strip().startswith('#'):
                return count
            count += 1
            if count == 1:
                name = line.strip()
                return count
            if count == 2:
                n_blades = int(line.split()[0])
                return count
            if count >= 9:
                r, c, beta = map(float, line.split()[:3])
                stations.append(r)
                chords.append(c)
                twists.append(beta)
                return count
            return count

        with open(file_path, 'r') as file:
            valid_line_count = 0
            for lin in file:
                valid_line_count = parse_line(lin, valid_line_count)

        stations_array = np.array(stations)
        chords_array = np.array(chords)
        twists_array = np.array(twists)

        new_prop = cls(
            n_blades=n_blades,
            stations=stations_array,
            chords=chords_array,
            twists=twists_array,
            airfoils=airfoils,
            name=name
        )

        return new_prop

    @classmethod
    def from_pickle(cls, file_path, rename: str | None = None) -> Self:
        with open(file_path, 'rb') as file:
            prop = pickle.load(file)
        if isinstance(prop, dict):
            loaded_prop = cls(
                n_blades=prop['n_blades'],
                stations=prop['stations'],
                chords=prop['chords'],
                twists=prop['twists'],
                offsets_x=prop['offsets_x'],
                offsets_z=prop['offsets_z'],
                thickness=prop['thickness'],
                name=prop['name'],
            )
            airfoils = [(station, Airfoil(coords)) for (station, coords) in
                        zip(prop['airfoils_stations'], prop['airfoils_airfoils'])]
            loaded_prop._airfoils = airfoils
            loaded_prop._hub_dimaeter = prop['hub_dimaeter']
            loaded_prop._hub_thickness = prop['hub_thickness']
            loaded_prop._hub_thickness_scale = prop['hub_thickness_scale']
            loaded_prop._hub_twist = prop['hub_twist']
            loaded_prop._hub_n_stations = prop['hub_n_stations']
            loaded_prop._density_points = prop['density_points']
            loaded_prop._weight = prop['weight']
            prop: Self = loaded_prop

        assert not isinstance(prop, dict)
        if rename is not None:
            prop.rename(rename)
        return prop

    @classmethod
    def from_prop_interpolation(cls, props: list[Propeller], weights: list | np.ndarray | None = None,
                                num_stations: int | None = None,
                                n_blades: int = 2, bunching_strength: tuple[float, float] = (1.0, 2.0),
                                name: str = 'Interpolated Propeller') -> Self:
        if weights is None:
            weights = np.ones(len(props))

        if num_stations is None:
            num_stations = max([len(prop.stations) for prop in props])

        weights = np.array(weights / np.sum(weights))
        new_diameter = np.mean([prop.diameter for prop in props])

        stations = np.zeros(num_stations)
        chords = np.zeros(num_stations)
        twists = np.zeros(num_stations)
        offsets_x = np.zeros(num_stations)
        offsets_z = np.zeros(num_stations)
        thickness = np.zeros(num_stations)

        all_airfoils = []
        for prop, weight in zip(props, weights):
            new_prop = copy.deepcopy(prop)
            new_prop.scale(float(new_diameter))
            new_prop.refine_stations(num_stations, bunching_strength)
            stations += weight * new_prop.stations
            chords += weight * new_prop.chords
            twists += weight * new_prop.twists
            offsets_x += weight * new_prop.offsets_x
            offsets_z += weight * new_prop.offsets_z
            thickness += weight * new_prop.thickness
            all_airfoils.append(
                new_prop.get_interpolated_airfoils(new_prop.stations))

        all_airfoils = np.array(all_airfoils, dtype='object')
        airfoils = [Airfoil.from_foil_interpolation(
            foils, weights) for foils in all_airfoils.T]
        airfoils = list(zip(stations, airfoils))
        final_prop = cls(n_blades, stations, chords, twists, airfoils,
                         offsets_x, offsets_z, thickness, name=name)
        return final_prop

    @classmethod
    def from_shape_parameters(cls, chord_parameters: np.ndarray, twist_parameters: np.ndarray,
                              diameter: float, num_stations: int = 30, n_blades: int = 2,
                              inner_diameter: float | None = None, bunching_strength: tuple[float, float] = (1.0, 2.0),
                              name: str = 'Unnamed Propeller') -> Self:
        inner_diameter = 0.2 * diameter if inner_diameter is None else inner_diameter
        stations = cls._make_stations_distribution(
            diameter, inner_diameter, num_stations+1, bunching_strength)[:-1]
        r = stations
        t = (2 * r / diameter)
        stations *= diameter / (2 * stations[-1])
        n_chords = len(chord_parameters)
        n_twists = len(twist_parameters)

        def legendre_poly(_t):
            return np.array([
                np.ones_like(t),
                2 * _t - 1,
                1.5 * (2 * _t - 1) ** 2 - 0.5,
                -3.0 * _t + 2.5 * (2 * _t - 1) ** 3 + 1.5,
                4.375 * (2 * _t - 1) ** 4 - 3.75 * (2 * _t - 1) ** 2 + 0.375,
                3.75 * _t + 7.875 * (2 * _t - 1) ** 5 -
                8.75 * (2 * _t - 1) ** 3 - 1.875
            ])

        L = legendre_poly(t**0.5)
        chords = np.dot(chord_parameters,
                        L[:n_chords])*np.sqrt(1 - np.sqrt(t))/10
        twists = np.degrees(
            np.arctan(np.dot(twist_parameters, L[:n_twists]) * (10*t)**-1))
        return cls(n_blades, stations, chords, twists, name=name)

    @staticmethod
    def database_list() -> list[str]:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        database_path = os.path.join(script_dir, 'apc_database', 'geometry')
        database_list = []
        for filename in os.listdir(database_path):
            database_list.append(filename.replace('-PERF.PE0', ''))
        return database_list

    @staticmethod
    def _make_stations_distribution(
            diameter: float, inner_diameter: float, num_stations: int,
            bunching_strength: tuple[float, float] = (1.0, 2.0),
    ) -> np.ndarray:
        dydx0 = 1 / bunching_strength[0]
        dydx1 = 1 / bunching_strength[1]
        x = np.linspace(0, 1, num_stations)
        placement_function = CubicHermiteSpline(
            x=np.array([0.0, 1.0]),
            y=np.array([0.0, 1.0]),
            dydx=np.array([dydx0, dydx1]),
        )
        blade_length = (diameter - inner_diameter) / 2
        stations = inner_diameter / 2 + placement_function(x) * blade_length
        return stations

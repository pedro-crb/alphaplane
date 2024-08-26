from __future__ import annotations

from typing import Optional, Self
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from scipy.interpolate import interp1d
from matplotlib import cm, colors

import matplotlib.pyplot as plt
import numpy as np


class Curve2d:
    def __init__(self, coords: np.ndarray, name: str = 'Unnamed Curve') -> None:
        coords = np.array(coords) + 0.
        self._name: str = name
        self._coords: np.ndarray = np.array(coords)
        self._area: Optional[float] = None
        self._perimeter: Optional[float] = None
        self._edge_sizes: Optional[np.ndarray] = None
        self._arclength: Optional[np.ndarray] = None
        self._edge_angles: Optional[np.ndarray] = None
        self._curvature: Optional[np.ndarray] = None
        self._normals: Optional[np.ndarray] = None
        self._edge_normals: Optional[np.ndarray] = None
        self._tangents: Optional[np.ndarray] = None
        self._edge_tangents: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return f"{self.num_pts}-point {self.__class__.__name__}')"

    def __eq__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return np.array_equal(self.coords, other.coords)

    def __hash__(self) -> int:
        return hash(self.coords.tobytes())

    @property
    def coords(self) -> np.ndarray:
        return np.array(self._coords) + 0.

    @property
    def num_pts(self) -> int:
        return len(self.coords)

    @property
    def is_closed(self) -> bool:
        return np.array_equal(self.coords[0], self.coords[-1])

    @property
    def coords_x(self) -> np.ndarray:
        return np.array(self._coords[:, 0]) + 0.

    @property
    def coords_y(self) -> np.ndarray:
        return np.array(self._coords[:, 1]) + 0.

    @property
    def area(self) -> float:
        if self._area is None:
            coords = self.coords
            x, y = coords[:, 0], coords[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            self._area = float(area)
        return self._area

    @property
    def edge_sizes(self) -> np.ndarray:
        if self._edge_sizes is None:
            self._edge_sizes = np.sqrt(np.diff(self.coords_x) ** 2 + np.diff(self.coords_y) ** 2)
        return np.array(self._edge_sizes) + 0.

    @property
    def arclength(self) -> np.ndarray:
        if self._arclength is None:
            self._arclength = np.concatenate([np.array([0.0]), np.cumsum(self.edge_sizes)])
        return np.array(self._arclength) + 0.

    @property
    def perimeter(self) -> float:
        if self._perimeter is None:
            self._perimeter = float(np.max(self.arclength))
        return self._perimeter

    @property
    def edge_angles(self) -> np.ndarray:
        if self._edge_angles is None:
            dx = np.diff(self.coords[:, 0])
            dy = np.diff(self.coords[:, 1])
            orientations = np.degrees(np.arctan2(dy, dx))
            anglescw = np.mod(np.diff(orientations), 360)
            anglesccw = anglescw - 360
            angles = np.where(np.abs(anglescw) < np.abs(anglesccw), anglescw, anglesccw)
            angles = np.concatenate([angles[[0]], angles, angles[[-1]]])
            self._edge_angles = angles
        return np.array(self._edge_angles) + 0.

    @property
    def curvature(self) -> np.ndarray:
        if self._curvature is None:
            x, y = self.coords[:, 0], self.coords[:, 1]
            dx = np.gradient(x)
            dy = np.gradient(y)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = (dx * ddy - dy * ddx) / np.power(dx ** 2 + dy ** 2, 1.5)
            self._curvature = curvature
        return np.array(self._curvature) + 0.

    @property
    def normals(self) -> np.ndarray:
        if self._normals is None:
            x, y = self.coords[:, 0], self.coords[:, 1]
            dx = np.gradient(x)
            dy = np.gradient(y)
            normals = np.array([dy, -dx])
            norm_lengths = np.sqrt(normals[0] ** 2 + normals[1] ** 2)
            normals = (normals / norm_lengths).T  # Normalize the normals
            self._normals = normals
        return np.array(self._normals) + 0.

    @property
    def edge_normals(self) -> np.ndarray:
        if self._edge_normals is None:
            x, y = self.coords[:, 0], self.coords[:, 1]
            dx = np.diff(x)
            dy = np.diff(y)
            normals = np.array([dy, -dx])
            norm_lengths = np.sqrt(normals[0] ** 2 + normals[1] ** 2)
            normals = (normals / norm_lengths).T  # Normalize the normals
            self._edge_normals = normals
        return np.array(self._edge_normals) + 0.

    @property
    def tangents(self) -> np.ndarray:
        if self._tangents is None:
            x, y = self.coords[:, 0], self.coords[:, 1]
            dx = np.gradient(x)
            dy = np.gradient(y)
            tangents = np.array([dx, dy])
            norm_lengths = np.sqrt(tangents[0] ** 2 + tangents[1] ** 2)
            tangents = (tangents / norm_lengths).T
            self._tangents = tangents
        return np.array(self._tangents) + 0.

    @property
    def edge_tangents(self) -> np.ndarray:
        if self._edge_tangents is None:
            x, y = self.coords[:, 0], self.coords[:, 1]
            dx = np.diff(x)
            dy = np.diff(y)
            tangents = np.array([dx, dy])
            norm_lengths = np.sqrt(tangents[0] ** 2 + tangents[1] ** 2)
            tangents = (tangents / norm_lengths).T
            self._edge_tangents = tangents
        return np.array(self._edge_tangents) + 0.

    @property
    def x_max(self) -> float:
        return np.max(self.coords[:, 0])

    @property
    def x_min(self) -> float:
        return np.min(self.coords[:, 0])

    @property
    def y_max(self) -> float:
        return np.max(self.coords[:, 1])

    @property
    def y_min(self) -> float:
        return np.min(self.coords[:, 1])

    @property
    def name(self) -> str:
        if self._name is None:
            self._name = 'Unnamed Curve'
        return self._name

    def merge_points(self, tolerance: float = 1e-4,
                     preserved_indices: Optional[list[int]] = None) -> None:
        if preserved_indices is None:
            preserved_indices = []

        indices = [0]
        distance = 0
        for i, s in enumerate(self.edge_sizes):
            distance += s
            if distance > tolerance:
                indices.append(i+1)
                distance = 0
            if distance < tolerance and (i == self.num_pts or i+1 in preserved_indices):
                if i > 0:
                    indices.pop()
                indices.append(i+1)
                distance = 0

        coords = self.coords[indices]
        self.set_coords(coords)

    def rename(self, new_name: str) -> None:
        self._name = new_name

    def translated(self, vector_xy: np.ndarray | list[float]) -> Curve2d:
        coords = self.coords + np.array(vector_xy)
        return Curve2d(coords)

    def scaled(self, scale_xy: np.ndarray | list[float] | float,
               center_xy: Optional[np.ndarray | list[float]] = None) -> Curve2d:
        if center_xy is None:
            center_xy = np.array([0.0, 0.0])

        scale_xy = np.array(scale_xy).reshape((-1,))
        if len(scale_xy) == 1:
            scale_xy = np.concatenate([scale_xy, scale_xy])

        center_xy = np.array(center_xy)
        coords = (self.coords - center_xy) * scale_xy + center_xy
        return Curve2d(coords)

    def rotated(self, angle_deg: float,
                center_xy: Optional[np.ndarray | list[float]] = None
                ) -> Curve2d:
        """returns curve rotated clockwise"""
        if center_xy is None:
            center_xy = np.array([0.0, 0.0])

        theta = -np.radians(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        coords = np.dot(self.coords - center_xy, R.T) + center_xy
        return Curve2d(coords)

    def transformed(self, source: np.ndarray, destination: np.ndarray) -> Curve2d:
        """Applies rotation, scaling, and translation to bring source points to destination points"""
        def line_angle(line):
            diff = line[1] - line[0]
            return np.arctan2(diff[1], diff[0])

        angle_source = line_angle(source)
        angle_destination = line_angle(destination)
        angle_diff = np.degrees(angle_destination - angle_source)

        translation_vector = destination[0] - source[0]

        source_distance = np.linalg.norm(source[1] - source[0])
        destination_distance = np.linalg.norm(destination[1] - destination[0])
        scale_factor = float(destination_distance / source_distance) if source_distance != 0 else 1.0

        translated_curve = self.translated(translation_vector)
        scaled_curve = translated_curve.scaled(scale_factor, destination[0])
        transformed_curve = scaled_curve.rotated(-angle_diff, destination[0])

        return transformed_curve

    def close(self) -> None:
        if not self.is_closed:
            new_coords = np.concatenate((self.coords, self.coords[0]))
            self.set_coords(new_coords)
    
    def set_coords(self, new_coords: np.ndarray) -> None:
        name = self.name
        self._clear_properties()
        self._coords = np.array(new_coords)
        self._name = name

    def _clear_properties(self) -> None:
        for key in self.__dict__.keys():
            setattr(self, key, None)

    def plot(self, color: str = 'black',
             marker: str = '',
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

        coords = self.coords

        label = self.name if label is None else label

        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth, label=label)

        if show:
            ax.axis('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True)
            ax.set_title('Curve')
            if legend:
                ax.legend()
            plt.show()

        return fig, ax

    def plot_comb(self, data: np.ndarray | list[float],
                  scale_factor: float = 1.0,
                  label: str | None = None,
                  legend: bool | None = None,
                  colormap: str | colors.Colormap | None = None,
                  normalize_cmap: bool = True,
                  colorbar: bool = False,
                  show: bool = True,
                  arrow: bool = False,
                  spline_color: str = 'gray',
                  spline_width: float = 1.0,
                  supress_combs: bool = False,
                  supress_curve: bool = False,
                  fig_ax: tuple[Figure, Axes] | None = None
                  ) -> tuple[Figure, Axes]:
        assert len(data) == self.num_pts, "Data length must match the number of points in the curve."

        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))
        else:
            fig, ax = fig_ax

        if normalize_cmap:
            norm = colors.Normalize(vmin=min(data), vmax=max(data))
        else:
            norm = colors.Normalize(vmin=-1, vmax=1)

        mappable = None
        if isinstance(colormap, str):
            cmap = cm.get_cmap(colormap)
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        if isinstance(colormap, colors.Colormap):
            cmap = colormap
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

        spline_points = []
        for (i, (_data, normal)) in enumerate(zip(data, self.normals)):
            arrow_vector = normal * _data * scale_factor
            if _data < 0 and arrow:
                start_point = self.coords[i] - arrow_vector
                spline_point = start_point
            else:
                start_point = self.coords[i]
                spline_point = start_point + arrow_vector
            spline_points.append(spline_point)
            color = mappable.to_rgba(np.array(_data)) if mappable is not None else 'royalblue'
            if not supress_combs:
                ax.arrow(float(start_point[0]), float(start_point[1]),
                         arrow_vector[0], arrow_vector[1],
                         length_includes_head=True,
                         width=0.001,
                         head_width=np.clip(0.05 * arrow * abs(_data)*scale_factor, 0.001, 0.015),
                         color=color)
        spline_points = np.array(spline_points)

        zorder = None if supress_combs else -1
        ax.plot(spline_points[:, 0], spline_points[:, 1],
                color=spline_color, zorder=zorder, linewidth=spline_width)

        if colorbar:
            assert mappable is not None
            fig.colorbar(mappable, ax=ax, orientation='vertical', label=label)

        if not supress_curve:
            ax.plot(self.coords[:, 0], self.coords[:, 1], color='black')

        if show:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True)
            ax.axis('equal')
            ax.set_ylim((-0.3, 0.3))
            ax.set_xlim((-0.8, 1.5))
            if legend:
                ax.legend()
            plt.show()

        return fig, ax

    def plot_colorline(self, data: np.ndarray | list[float],
                       label: Optional[str] = None,
                       legend: bool = False,
                       colormap: Optional[str] = None,
                       show: bool = True,
                       fig_ax: Optional[tuple[Figure, Axes]] = None
                       ) -> tuple[Figure, Axes]:
        assert len(data) == self.num_pts, "Data length must match the number of points in the curve."

        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax

        cmap = plt.get_cmap(colormap)

        norm = colors.Normalize(min(data), max(data))
        _colors = cmap(norm(data))

        # Plot each segment of the curve with the corresponding color
        for i in range(self.num_pts - 1):
            x = self.coords[i:i + 2, 0]
            y = self.coords[i:i + 2, 1]
            color = _colors[i]
            ax.plot(x, y, color=color)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=label)

        if show:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.axis('equal')
            if legend:
                ax.legend()
            plt.show()

        return fig, ax

    @classmethod
    def from_curve_join(cls, coords1: np.ndarray, coords2: np.ndarray, num_samples: int = 10,
                        name: str = 'Unnamed Curve') -> Self:
        """fills a gap between two curves using a cubic spline"""
        sample_coords = np.vstack([coords1[-2:], coords2[:2]])
        distances = np.linalg.norm(np.diff(sample_coords, axis=0), axis=1)
        if distances[1] == 0:
            return cls(np.vstack([coords1, coords2]), name)
        tsamples = [-distances[0] / distances[1], 0.0, 1.0, 1 + distances[2] / distances[1]]
        splinex = interp1d(tsamples, sample_coords[:, 0], kind='cubic')
        spliney = interp1d(tsamples, sample_coords[:, 1], kind='cubic')
        tqueries = np.linspace(0.0, 1.0, num_samples + 2)[1:-1]
        fill_points = np.vstack([splinex(tqueries), spliney(tqueries)]).T
        joined_coords = np.vstack([coords1, fill_points, coords2])
        return cls(joined_coords, name)

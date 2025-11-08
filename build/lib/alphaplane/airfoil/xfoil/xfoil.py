from __future__ import annotations

import numpy as np
import warnings
import copy
import os

from typing import TYPE_CHECKING
from scipy.interpolate import interp1d

from alphaplane.airfoil.foil_condition import AirfoilCondition
from alphaplane.wrapper_tools.executable import Executable

if TYPE_CHECKING:
    from alphaplane.airfoil.airfoil import Airfoil, FlappedAirfoil


def run(airfoil: Airfoil,
        condition: AirfoilCondition,
        max_iter: int = 100,
        timeout_seconds: float = 2) -> tuple[np.ndarray, np.ndarray]:
    assert condition.is_alpha_polar, ('Multiple operating conditions were passed.\n'
                                      'The condition may only have multiple alphas')

    _airfoil = airfoil.deflected(float(condition.deflection), float(condition.hinge_position)) # type: ignore
    
    assert condition.alpha is not None
    alphas_remaining = list(condition.alpha)
    polar = []
    bl_data = []
    while alphas_remaining:
        new_condition = copy.deepcopy(condition)
        new_condition.alpha = alphas_remaining  # type: ignore
        _polar, _bl_data, alphas_remaining, _ = (
            run_attempt(_airfoil, new_condition, max_iter=max_iter, timeout_seconds=timeout_seconds)
        )
        polar.append(_polar)
        bl_data.append(_bl_data)

    polar = np.concatenate(polar)
    bl_data = np.concatenate(bl_data)

    confidences = np.ones_like(condition.alpha, dtype='float')
    polar = np.hstack((polar, np.atleast_2d(confidences).T))

    return polar, bl_data


def run_attempt(airfoil: FlappedAirfoil,
                condition: AirfoilCondition,
                max_iter: int = 100,
                timeout_seconds: float = 5
                ) -> tuple[np.ndarray, np.ndarray, list[float], BaseException | None]:
    assert not condition.is_deflected, ('Passing a deflection through the condition is not supported\n'
                                        'To run with flap deflection, a FlappedAirfoil must be passed.')

    assert condition.is_alpha_polar, ('Multiple operating conditions were passed.\n'
                                      'The condition may only have multiple alphas')

    alphas_remaining = copy.deepcopy(list(condition.alpha)) # type: ignore

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xfoil_exe_path = os.path.join(script_dir, 'xfoil.exe')
    executable = Executable(xfoil_exe_path, capture_output=True, timeout_seconds=timeout_seconds)

    with (executable):

        airfoil_dat_path = os.path.join(executable.temp_dir, 'airfoil.dat') # type: ignore
        airfoil.to_dat(airfoil_dat_path)
        polar_path = os.path.join(executable.temp_dir, 'polar.txt') # type: ignore
        bl_data_paths = []
        valid_indices = []

        executable.send_command(f"PLOP")
        executable.send_command(f"G F")
        executable.send_command(f"")
        executable.send_command(f"LOAD {airfoil_dat_path}")
        executable.send_command("OPER")
        executable.send_command(f"VISC {float(condition.reynolds):.2f}") # type: ignore
        executable.send_command(f"MACH {float(condition.mach):.4f}") # type: ignore
        executable.send_command("VPAR")
        executable.send_command(f"XTR {float(condition.xtr_top):.4f} {float(condition.xtr_bottom):.4f}") # type: ignore
        executable.send_command(f"N {float(condition.n_crit):.4f}") # type: ignore
        executable.send_command("")
        executable.send_command(f"ITER {max_iter}")
        executable.send_command(f"PACC")
        executable.send_command(f"{polar_path}")
        executable.send_command("")

        previous_alpha = None
        sent_init = False
        alphas = np.array(condition.alpha)
        alphas_run = []
        for (i, alpha) in enumerate(alphas):
            alphas_remaining.pop(0)
            alphas_run.append(alpha)
            if previous_alpha is not None and abs(alpha - previous_alpha) > 2 and not sent_init:
                executable.send_command(f"INIT")
            sent_init = False
            response = executable.send_command(
                command=f"ALFA {alpha:.4f}",
                wait_for=["Point added to stored polar", "VISCAL:  Convergence failed"],
                clear_buffer=True
            )
            if response is not None:
                response = response[(-min(300, len(response))):]
            else:
                response = 'NaN'
            if "Point added to stored polar" in response and \
                    "NaN" not in response and "VISCAL:  Convergence failed" not in response:
                executable.send_command(f"DUMP")
                bl_data_path = os.path.join(executable.temp_dir, f'dump{i}.txt') # type: ignore
                bl_data_paths.append(bl_data_path)
                executable.send_command(f"{bl_data_path}")
                valid_indices.append(i)
            elif response is None and executable.process.poll() is not None: # type: ignore
                break
            else:
                executable.send_command(f"INIT")
                sent_init = True
 
        if executable.process.poll() is None: # type: ignore
            try:
                executable.send_command("")
                executable.send_command("QUIT")
                executable.process.wait() # type: ignore
            except OSError:
                pass

        alphas_run = np.array(alphas_run)
        polar = parse_polar(polar_path, valid_indices, alphas_run)
        point_is_valid = ~np.array([any(np.isnan(p)) for p in polar])
        valid_indices = [i for (i, v) in zip(valid_indices, point_is_valid[valid_indices]) if v]
        valid_indices_bl = [i for (i, v) in enumerate(point_is_valid[valid_indices]) if v]
        polar[~point_is_valid] = np.full((np.sum(~point_is_valid), 6), np.nan) # type: ignore

        bl_data, cp_min, mcrit = parse_bl_data(bl_data_paths, valid_indices_bl, valid_indices, point_is_valid) # type: ignore
        polar = np.hstack((polar, cp_min, mcrit))

    return polar, bl_data, alphas_remaining, None


def parse_polar(file_path: str, valid_indices: list[bool], alphas: np.ndarray) -> np.ndarray:
    data = []
    with open(file_path, 'r') as lines:
        for line in lines:
            if line.strip().startswith('alpha'):
                break
        next(lines)
        for line in lines:
            if line.strip():
                row_data = line.split()
                if len(row_data) == 7:
                    data.append(np.array(row_data, dtype='float'))

    valid_alphas = alphas[valid_indices]
    data = np.atleast_2d(data)
    final_data = np.full((len(alphas), 6), np.nan)

    if data.size == 0 or len(valid_alphas) == 0:
        return final_data

    data_alphas = data[:, 0]
    data_indices = [min(range(len(data_alphas)), key=lambda _i: abs(data_alphas[_i] - b)) for b in valid_alphas]

    final_data[valid_indices] = data[:, 1:][data_indices]
    return final_data


def parse_bl_data(file_paths: list[str], valid_indices_bl: list[int],
                  valid_indices: list[int], point_is_valid: list[bool]
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = []
    Cp_min = []
    Mcrit = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            data.append(np.array([]))
            continue

        file_data = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#') or not line.strip():
                    continue
                parts = [line[:10], line[10:19], line[19:28], line[28:37], line[37:47], line[47:57], line[57:67],
                         line[67:]]
                values = [np.nan if '*' in part else float(part) for part in parts]
                file_data.append(values)
        if file_data:
            new_data = np.array(file_data)
            bl_Ue_Vinf = new_data[:, 3]
            bl_Cp = 1 - bl_Ue_Vinf ** 2
            new_cpmin = np.min(bl_Cp)
            max_vel_ratio = np.max(bl_Ue_Vinf)
            new_mcrit = 0.0 if max_vel_ratio == 0 else 1/max_vel_ratio
            new_data = np.vstack([new_data.T, [bl_Cp]]).T
        else:
            new_data = np.array([])
            new_cpmin = np.array([])
            new_mcrit = np.array([])
        data.append(new_data)
        Cp_min.append(new_cpmin)
        Mcrit.append(new_mcrit)

    final_data = np.full((len(point_is_valid), 9), None, dtype='object')
    final_Cp_min = np.full((len(point_is_valid), 1), np.nan)
    final_Mcrit = np.full((len(point_is_valid), 1), np.nan)

    if len(valid_indices_bl) > 0 and len(valid_indices) > 0:
        for i_final, i_bl in zip(valid_indices, valid_indices_bl):
            try:
                final_data[i_final] = list(data[i_bl].T)
            except ValueError:
                pass
        try:
            final_Cp_min[point_is_valid] = [[Cp_min[v]] for v in valid_indices_bl]
            final_Mcrit[point_is_valid] = [[Mcrit[v]] for v in valid_indices_bl]
        except ValueError:
            pass

    return final_data, np.array(final_Cp_min), np.array(final_Mcrit)


def interpolate_polar(alphas: np.ndarray, polar_data: np.ndarray, alphas_query: np.ndarray) -> np.ndarray:
    valid_indices = np.where(~np.isnan(polar_data[:, 0]))
    valid_alphas = alphas[valid_indices]
    valid_values = polar_data[valid_indices]
    if len(np.atleast_1d(alphas)) > 1 and len(valid_alphas) > 1:
        polar_interp = interp1d(valid_alphas, valid_values, axis=0, bounds_error=False)(alphas_query)
    else:
        polar_interp = polar_data

    return polar_interp


def validate_airfoil(airfoil: Airfoil) -> None:
    max_panel_angle = np.max(airfoil.edge_angles)
    if max_panel_angle > 40:
        warnings.warn(f'\nMaximum angle between panels is {max_panel_angle}')

    if airfoil.num_pts > 270:
        warnings.warn(f'number of airfoil points ({airfoil.num_pts} points for "{airfoil.name}") '
                      f'may be too high for XFOIL')

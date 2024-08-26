from __future__ import annotations

from typing import TYPE_CHECKING

import neuralfoil_standalone as nf
import numpy as np
import copy

if TYPE_CHECKING:
    from alphaplane.airfoil.airfoil import Airfoil
    from alphaplane.airfoil.foil_condition import AirfoilCondition


def run_from_airfoil(airfoil: Airfoil, condition: AirfoilCondition, model_size: str = 'large'
                     ) -> tuple[np.ndarray, np.ndarray]:

    current_condition = copy.deepcopy(condition)
    current_condition.set_defaults()

    if condition.is_sequence:
        params = current_condition.parameter_sequence()
    else:
        params = current_condition.parameter_grid().reshape(-1, current_condition.num_params())

    results = nf.get_aero_with_corrections(
        kulfan_parameters=airfoil.neuralfoil_parameters(),
        alpha=params[:, 0],
        Re=params[:, 1],
        mach=params[:, 2],
        n_crit=params[:, 7],
        xtr_upper=params[:, 5],
        xtr_lower=params[:, 6],
        model_size=model_size,
        control_surface_deflection=params[:, 3],
        control_surface_hinge_point=1-params[:, 4],
        wave_drag_foil_thickness=airfoil.thickness_max,
    )
    polar_data = np.array([
        results['CL'],
        results['CD'],
        results['CD'],
        results['CM'],
        results['Top_Xtr'],
        results['Bot_Xtr'],
        results['Cpmin_0'],
        results['mach_crit'],
        results['analysis_confidence']
    ]).T

    return polar_data, params


def run_from_kulfan_params(kulfan_params: dict, condition: AirfoilCondition, model_size: str = 'large'):
    current_condition = copy.deepcopy(condition)
    current_condition.set_defaults()

    if condition.is_sequence:
        params = current_condition.parameter_sequence()
    else:
        params = current_condition.parameter_grid().reshape(-1, current_condition.num_params())
    results = nf.get_aero_with_corrections(
        kulfan_parameters=kulfan_params,
        alpha=params[:, 0],
        Re=params[:, 1],
        mach=params[:, 2],
        n_crit=params[:, 7],
        xtr_upper=params[:, 5],
        xtr_lower=params[:, 6],
        model_size=model_size,
        control_surface_deflection=params[:, 3],
        control_surface_hinge_point=1-params[:, 4],
        wave_drag_foil_thickness=0.12,
    )
    return results

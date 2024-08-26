from __future__ import annotations

from typing import TYPE_CHECKING, Union

import copy
import warnings
import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator

from alphaplane.propeller.prop_data import PropellerData
from alphaplane.airfoil.foil_condition import AirfoilCondition
from alphaplane.propeller.prop_condition import PropellerCondition

if TYPE_CHECKING:
    from alphaplane.airfoil import Airfoil
    from alphaplane.propeller.propeller import Propeller


POLAR_COLUMNS = ['J', 'rpm', 'CT', 'CP']


class PropellerAnalysis:
    def __init__(self, propeller: Propeller) -> None:
        from alphaplane.propeller.propeller import Propeller
        self.propeller: Propeller = propeller
        self.data: PropellerData = PropellerData()
        # stored airfoil data
        self._analyzed_airfoils: list[tuple[float, Airfoil]] | None = None
        self._airfoil_data_interpolator: RegularGridInterpolator | None = None
        self._airfoil_condition: AirfoilCondition | None = None

    @property
    def condition(self):
        return self.data.condition

    def inertial_load(self, rpm) -> np.ndarray:
        weights = self.propeller.estimate_weights()
        r = np.array([w[0] for w in weights])
        m = np.array([w[1] for w in weights])
        w = 2 * np.pi * rpm / 60
        centripetal = m * w ** 2 * r
        load = np.cumsum(centripetal[::-1])[::-1]
        load = np.append(load, 0.0)
        loads = np.array(zip(r, load))
        return loads

    def polars_from_qprop(self, condition: PropellerCondition, density: float = 1.225, dyn_viscosity: float = 1.78e-5,
                          speed_of_sound: float = 340.0, output_unstructured: bool = False, max_iter: int = 20):
        if condition.is_sequence:
            conditions = condition.parameter_sequence()
        else:
            conditions = condition.parameter_grid().reshape(-1, 2)

        CT = []
        CP = []
        for c in conditions:
            current_condition = PropellerCondition(*c)
            prop_results, _ = self.run_qprop_point(current_condition, density, dyn_viscosity, speed_of_sound, max_iter)
            CT.append(prop_results['CT'])
            CP.append(prop_results['CP'])

        if condition.is_sequence or output_unstructured:
            new_points = np.hstack((conditions, CT, CP))
            self.data.add_unstructured_data(new_points)
        else:
            grid = condition.parameter_grid()
            new_points = np.vstack((CT, CP)).T.reshape(*grid.shape[:-1], 2)
            self.data.set_structured_data(new_points, condition)

    def run_qprop_point(self, condition: PropellerCondition, density: float = 1.225, dyn_viscosity: float = 1.78e-5,
                        speed_of_sound: float = 340.0, max_iter: int = 20) -> tuple[dict[str, float], pd.DataFrame]:
        """
        Propeller model based on:
        [1] Mark Drela, MIT Aero & Astro, QPROP Formulation, June 2006.
        """

        assert condition.is_single_point, 'Only a single point can be calculated here'
        assert condition.J is not None and condition.rpm is not None
        J = np.array(condition.J)
        rpm = np.array(condition.rpm)

        V = J * rpm * self.propeller.diameter / 60
        omega = 2 * np.pi * rpm / 60
        r = self.propeller.stations
        r[-1] = (r[-2] + 9 * r[-1]) / 10
        beta = np.radians(self.propeller.twists)
        rho = density
        c = self.propeller.chords
        mu = dyn_viscosity
        a = speed_of_sound
        R = self.propeller.radius
        B = self.propeller.n_blades

        if self._airfoil_data_interpolator is None:
            warnings.warn('\n\nNo airfoil data is present in the propeller object. '
                          'Calculating airfoil using default settings...\n'
                          'To avoid this warning, run analyze_airfoil with desired conditions'
                          'before calculating the propeller\n')
            self.analyze_airfoils(
                stations=6,
                condition=AirfoilCondition(
                    alpha=np.arange(-20, 20, 1),
                    reynolds=[3e4, 5e4, 7e4, 1e5, 1.5e5, 2e5, 3e5, 5e5, 7e5, 1e6, 1.5e6, 2e6],
                    mach=[0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.05, 1.2, 1.5, 2, 3],
                    xtr_top=0.0,
                    xtr_bottom=0.0,
                )
            )
        assert self._airfoil_data_interpolator is not None
        assert self._airfoil_condition is not None
        assert self._airfoil_condition.reynolds is not None
        assert self._airfoil_condition.mach is not None
        analyzed_stations = self._airfoil_data_interpolator.grid[0]
        min_r = np.min(analyzed_stations)
        max_r = np.max(analyzed_stations)
        min_reynolds = np.min(self._airfoil_condition.reynolds)
        max_reynolds = np.max(self._airfoil_condition.reynolds)
        min_mach = np.min(self._airfoil_condition.mach)
        max_mach = np.max(self._airfoil_condition.mach)
        query_others = np.squeeze(np.array(self._airfoil_condition.parameter_list()[3:]))
        query_others = np.tile(query_others, (len(r), 1))

        psi = np.radians(10 * np.ones_like(r))
        RE_EXP = 0.4
        RE_CORRECTION_MAX = 1.05
        alpha = np.array([])
        CL = np.array([])
        CD = np.array([])
        W_a = np.array([])
        W_t = np.array([])
        Re_real = np.array([])
        Gamma = np.array([])
        lamb = np.array([])
        Ma_real = np.array([])
        for _ in range(max_iter):
            U_a = V
            U_t = omega * r
            U = np.sqrt(U_a ** 2 + U_t ** 2)
            W_a = 0.5 * U_a + 0.5 * U * np.sin(psi)
            W_t = 0.5 * U_t + 0.5 * U * np.cos(psi)
            W = np.sqrt(W_a ** 2 + W_t ** 2)
            alpha = beta - np.arctan(W_a / W_t)
            v_tangent = U_t - W_t
            Re_real = rho * W * c / mu
            Re = np.clip(Re_real, min_reynolds, max_reynolds)
            Ma_real = W / a
            Ma = np.clip(Ma_real, min_mach, max_mach)
            lamb = (r / R) * (W_a / W_t)
            f = (B / 2) * (1 - r / R) / lamb
            F = (2 / np.pi) * np.arccos(np.exp(-f))
            Gamma = (4 * np.pi * r * v_tangent * F / B) * np.sqrt(1 + (4 * lamb * R / (np.pi * B * r)) ** 2)
            interp_query = np.vstack((np.clip(r, min_r, max_r), np.degrees(alpha), Re, Ma, query_others.T)).T
            interp_data = self._airfoil_data_interpolator(interp_query)
            Re_correction = (Re_real / Re) ** RE_EXP
            Re_correction = np.where(Re_correction < RE_CORRECTION_MAX, Re_correction, RE_CORRECTION_MAX)
            CL = interp_data[:, 0]
            CLa = interp_data[:, 9]
            CD = interp_data[:, 1]
            CL = np.where(np.isnan(CL), np.sin(2 * alpha), CL) * Re_correction
            CLa = np.clip(np.where(np.isnan(CLa), 0.0, CLa), -1.0, 10.0) * Re_correction
            CD = np.where(np.isnan(CD), 2 * np.sin(alpha) ** 2, CD) / Re_correction
            Res = Gamma - 0.5 * W * c * CL

            dW_a = 0.5 * U * np.cos(psi)
            dW_t = -0.5 * U * np.sin(psi)
            dv_t = -dW_t
            dlamb = (r / R) * (dW_a / W_t - W_a * (dW_t / W_t ** 2))
            df = -((B / 2) * (1 - r / R) / lamb ** 2) * dlamb
            dF = (2 / np.pi) / np.sqrt(1 - np.exp(-f) ** 2) * np.exp(-f) * df
            dG_helper = (lamb * dlamb * (4 * R / (np.pi * B * r)) ** 2) / np.sqrt(
                1 + (4 * lamb * R / (np.pi * B * r)) ** 2)
            dVtF = dv_t * F + dF * v_tangent
            dG_const = 4 * np.pi * r / B
            dG1 = dVtF * np.sqrt(1 + (4 * lamb * R / (np.pi * B * r)) ** 2)
            dG2 = v_tangent * F * dG_helper
            dGamma = dG_const * (dG1 + dG2)
            dWa_over_Wt = dW_a * (1 / W_t) - W_a * dW_t * (1 / W_t) ** 2
            dalpha = -(1 / (1 + (W_a / W_t) ** 2)) * dWa_over_Wt
            dW = (1 / W) * (W_a * dW_a + W_t * dW_t)
            dRe = rho * dW * c / mu
            dCL1 = CLa * dalpha
            dCL2 = np.where(Re_correction < RE_CORRECTION_MAX, CL * (RE_EXP / Re) * dRe, 0.0)
            dCL = dCL1 + dCL2
            dRes = dGamma - 0.5 * c * (W * dCL + CL * dW)

            psi -= (Res / dRes)
            psi = np.clip(psi, -np.pi / 2, np.pi / 2)
            if np.all(np.abs(Res) < 1e-8):
                break

        r[-1] = R

        dTdr = rho * B * Gamma * (np.maximum(W_t - (CD / CL) * W_a, 0.0))

        dQdr = rho * B * Gamma * r * (W_a + (CD / CL) * W_t)

        T = np.trapz(dTdr, r)
        Q = np.trapz(dQdr, r)
        prop_results = {
            'CT': float(T / (rho * (rpm / 60) ** 2 * (2 * R) ** 4)),
            'CP': float(2 * np.pi * Q / (rho * (rpm / 60) ** 2 * (2 * R) ** 5)),
            'thrust': T,
            'torque': Q
        }

        blade_results = pd.DataFrame(
            np.array([r, c, np.degrees(beta), np.degrees(alpha), CL, CD, W_a,
                      Re_real, Ma_real, lamb, dTdr, dQdr]).T,
            columns=['station', 'chord', 'twist', 'alpha', 'CL', 'CD', 'Wa',
                     'reynolds', 'mach', 'adv_wake', 'dTdr', 'dQdr']
        )

        return prop_results, blade_results

    def polars_from_APC(self, file_path: str) -> None:
        data = []

        def parse_polar(line, _rpm):
            if line.strip().startswith('PROP RPM = '):
                rpm_line_parts = line.split('=')
                _rpm[0] = float(rpm_line_parts[1])

            try:
                parts = [float(value) for value in line.split()]
            except ValueError:
                parts = []

            if len(parts) == 15:
                J = parts[1]
                CT = parts[3]
                CP = parts[4]
                data.append([J, _rpm[0], CT, CP])

        with open(file_path, 'r') as file:
            rpm = [0]
            for lin in file.readlines():
                parse_polar(lin, rpm)

        self.data.add_unstructured_data(np.array(data))
        self.data.regularize_unstructured_points()

    def get_points(self, condition: PropellerCondition | None = None, datasets='all'
                   ) -> pd.DataFrame:
        polar = self.data.get_points(condition, datasets)
        polar_df = pd.DataFrame(polar, columns=POLAR_COLUMNS)
        return polar_df

    def get_interpolated(self, condition: PropellerCondition, method='auto') -> pd.DataFrame:
        condition = copy.deepcopy(condition)

        def build_query(_condition: PropellerCondition) -> np.ndarray:
            if _condition.is_sequence:
                _query = np.ascontiguousarray(_condition.parameter_sequence())
            else:
                _query = np.ascontiguousarray(_condition.parameter_grid().reshape(-1, 2))
            return _query

        if (method == 'auto' and self.data.has_unstructured) or method == 'unstructured':
            interpolator = self.data.interpolate_general
            assert self.data.condition_unstructured is not None
            condition.fill_with_other(self.data.condition_unstructured)
        else:
            assert self.data.condition_structured is not None
            interpolator = self.data.interpolate_structured
            condition.fill_with_other(self.data.condition_structured)

        query = build_query(condition)
        data = interpolator(query)
        data_df = pd.DataFrame(data, columns=POLAR_COLUMNS)
        return data_df

    def analyze_airfoils(self, stations: list[float] | np.ndarray | int | None = None,
                         condition: AirfoilCondition | None = None,
                         method: str = 'neuralfoil', show_progress: bool = False, **kwargs) -> None:
        skip_analysis = False
        if isinstance(stations, int):
            analyzed_stations = np.linspace(self.propeller.stations[0], self.propeller.stations[-1], stations)
            analyzed_airfoils = self.propeller.get_interpolated_airfoils(analyzed_stations)
        elif isinstance(stations, Union[np.ndarray, list]):
            analyzed_stations = np.array(stations, dtype=float)
            analyzed_airfoils = self.propeller.get_interpolated_airfoils(analyzed_stations)
        elif stations is None:
            analyzed_stations, analyzed_airfoils = zip(*self.propeller.airfoils)
            assert isinstance(analyzed_airfoils, tuple)
        else:
            raise ValueError("A value of invalid type was passed for argument 'stations'")

        if condition is None:
            conditions = [foil.analysis.condition for foil in analyzed_airfoils]
            if all([cond == conditions[0] for cond in conditions]):
                condition = conditions[0]
            else:
                condition = AirfoilCondition.from_intersection(conditions)
            skip_analysis = True
        assert not condition.is_sequence, "Can't analyze a sequence of conditions for a propeller"
        check_conditions = [
            ((condition.deflection is None) or len(condition.deflection) == 0
             or (len(condition.deflection) == 1 and condition.deflection[0] == 0.0)),
            condition.hinge_position is None or len(condition.hinge_position) <= 1,
            condition.xtr_top is None or len(condition.xtr_top) <= 1,
            condition.xtr_bottom is None or len(condition.xtr_bottom) <= 1,
            condition.n_crit is None or len(condition.n_crit) <= 1,
        ]
        assert all(check_conditions), 'Multiple parameters are allowed only for alpha, reynolds, and mach'

        if not skip_analysis:
            for i, (station, foil) in enumerate(zip(analyzed_stations, analyzed_airfoils)):
                foil.analysis.data.clear_all()
                if show_progress:
                    print(f'\rRunning blade sections: ({i}/{len(analyzed_airfoils)})', end='')
                assert condition is not None
                foil.analysis.run(condition=condition, method=method, **kwargs)
            if show_progress:
                print(f'\rRunning blade sections: ({len(analyzed_airfoils)}/{len(analyzed_airfoils)})')
            new_analyzed_airfoils = list(zip(analyzed_stations, analyzed_airfoils))
            if self._analyzed_airfoils is None:
                self._analyzed_airfoils = new_analyzed_airfoils
            else:
                self._analyzed_airfoils = (
                    self._analyzed_airfoils + new_analyzed_airfoils
                )
            self._analyzed_airfoils.sort(key=lambda f: f[0])

        foil_data = []
        for foil in analyzed_airfoils:
            data = foil.analysis.data.data_structured
            alpha_derivative = (180 / np.pi) * foil.analysis.data.grad_structured(0)
            final_data = np.concatenate((data, alpha_derivative), axis=-1)[..., :(data.shape[-1] + 1)]
            foil_data.append(final_data)

        analyzed_condition = analyzed_airfoils[0].analysis.data.condition_structured
        self._airfoil_condition = analyzed_condition
        interpolator_grid = (np.array(analyzed_stations, dtype=float), *analyzed_condition.parameter_list())
        data_ndpolator = RegularGridInterpolator(
            interpolator_grid,
            np.array(foil_data),
            bounds_error=False
        )
        self._airfoil_data_interpolator = data_ndpolator

    def clear_analyzed_airfoils(self):
        self._analyzed_airfoils = None

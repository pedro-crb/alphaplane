from __future__ import annotations
from typing import Literal

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

from alphaplane.propeller.propeller import Propeller
from alphaplane.propeller.prop_condition import PropellerCondition
from alphaplane.propulsion_system import ElectricMotor, Battery, SpeedController


class PropulsionSystem:
    def __init__(self, propeller: Propeller,
                 motor: ElectricMotor,
                 battery: Battery,
                 speed_controller: SpeedController = SpeedController(),
                 throttle_efficiency_full: float = 0.98,
                 throttle_efficiency_half: float = 0.91,
                 throttle_efficiency_zero: float = 0.50,
                 ) -> None:
        self.motor: ElectricMotor = motor
        self.battery: Battery = battery
        self.speed_controller: SpeedController = speed_controller
        self.propeller: Propeller = propeller
        self.throttle_efficiency = PchipInterpolator(
            np.array([0.0, 0.5, 1.0, 1.1]),
            np.array([throttle_efficiency_zero,
                      throttle_efficiency_half,
                      throttle_efficiency_full, throttle_efficiency_full])
        )
        self.throttle_efficiency_derivative = self.throttle_efficiency.derivative()

    @property
    def is_valid(self) -> bool:
        is_valid = True

        if self.motor.max_volts is not None and not np.isnan(self.motor.max_volts):
            is_valid = is_valid and (self.battery.volts < self.motor.max_volts)

        if self.speed_controller.max_volts is not None:
            is_valid = is_valid and (self.battery.volts < self.speed_controller.max_volts)

        return is_valid

    @property
    def weight(self) -> float:
        weights = [self.motor.weight, self.battery.weight, self.speed_controller.weight, self.propeller.weight]
        weight = 0.0
        for w in weights:
            weight += w if w is not None else 0.0
        return weight

    @property
    def max_amps(self) -> float | None:
        max_amps = np.inf
        amps_values = [
            self.battery.max_amps,
            self.speed_controller.max_amps,
            self.motor.max_amps
        ]
        for amps in amps_values:
            if amps is not None:
                max_amps = np.minimum(max_amps, amps)

        watts_values = [
            self.speed_controller.max_watts,
            self.motor.max_watts
        ]
        for watts in watts_values:
            if watts is not None:
                max_amps = np.minimum(max_amps, watts / self.battery.volts)

        if np.isinf(max_amps):
            max_amps = None

        return max_amps

    def run(self, velocity: float, parameter: str,
            value: float, density: float = 1.225,
            dyn_viscosity: float = 1.78e-5, speed_of_sound: float = 340.0,
            propeller_method: Literal['qprop', 'polar'] = 'qprop',
            max_iter: int = 20, ignore_limits: bool = False
            ) -> tuple[dict, list[dict], pd.DataFrame]:

        if not self.is_valid and not ignore_limits:
            return {}, [{}], pd.DataFrame()

        solver = self._make_solution_function(parameter)

        rpmmin = None
        rpmmax = None
        if propeller_method == 'polar':
            # Here we access directly from the stored data in the propeller object to make it run faster
            CP = self.propeller.analysis.data.data_structured[0, 0, 1]
            rpmmin = np.min(self.propeller.analysis.data.condition.rpm) # type: ignore
            rpmmax = np.max(self.propeller.analysis.data.condition.rpm) # type: ignore
        elif propeller_method == 'qprop':
            CP = self.propeller.analysis.run_qprop_point(
                PropellerCondition(J=0.2, rpm=5000),
                density, dyn_viscosity, speed_of_sound
            )[0]['CP']
        else:
            raise ValueError('Invalid Propeller Method')

        rpm_crit = 120 * np.pi * speed_of_sound / self.propeller.radius
        system_data = None
        efficiencies = None
        blade_data = None
        converged = False
        i = 0
        run_count = 0
        while i < max_iter and run_count < 3:
            throttle, amps = solver.solve(value, density, abs(CP))
            system_data, efficiencies = self._run_system(throttle, amps)
            rpm = system_data['rpm']
            D = self.propeller.diameter
            J = velocity / ((rpm / 60) * D)
            if propeller_method == 'polar':
                rpm_interp = np.clip(rpm, rpmmin, rpmmax)
                interp_result = self.propeller.analysis.data.interpolate_structured(
                    np.atleast_2d([J, rpm_interp])
                )
                propeller_data = dict(zip(['J', 'rpm', 'CT', 'CP'], interp_result[0]))
                blade_data = pd.DataFrame()
            elif propeller_method == 'qprop':
                qprop_data = self.propeller.analysis.run_qprop_point(
                    PropellerCondition(J, rpm), density, dyn_viscosity, speed_of_sound
                )
                propeller_data = qprop_data[0]
                blade_data = qprop_data[1]
            else:
                raise ValueError('Invalid Propeller Method')

            CQ = propeller_data['CP']/(2*np.pi)
            CT = propeller_data['CT']
            system_data['prop_torque'] = (rpm/60)**2 * D**5 * density * CQ
            system_data['thrust'] = (rpm/60)**2 * D**4 * density * CT
            system_data['CP'] = propeller_data['CP']
            system_data['CT'] = CT
            system_data['CQ'] = CQ

            if abs(CP - system_data['CP']) < 1e-5:
                if ignore_limits:
                    converged = True
                    break

                if rpm > rpm_crit + 1e-1:
                    solver = self._make_solution_function('rpm')
                    value = rpm_crit
                    i = 0
                    run_count += 1
                elif self.max_amps is not None and system_data['amps'] > self.max_amps + 1e-2:
                    solver = self._make_solution_function('amps')
                    value = self.max_amps
                    i = 0
                    run_count += 1
                elif system_data['throttle'] > 1.0 + 1e-3:
                    solver = self._make_solution_function('throttle')
                    value = 1.0
                    i = 0
                    run_count += 1
                else:
                    converged = True
                    break

            if np.isnan(system_data['CP']):
                break

            CP = system_data['CP']
            i += 1

        if converged and system_data is not None and efficiencies is not None and blade_data is not None:
            return system_data, efficiencies, blade_data
        else:
            return {}, [{}], pd.DataFrame()

    def _run_system(self, throttle: float, amps: float) -> tuple[dict[str, float], list[dict[str, float | str]]]:
        volts = throttle * self.battery.volts
        efficiencies = [{'volts': volts, 'amps': amps, 'efficiency': 1.0, 'part': 'initial'}]

        battery_dropoff = amps * self.battery.resistance
        volts_1 = volts - battery_dropoff
        efficiencies.append({'volts': volts_1, 'amps': amps, 'efficiency': volts_1 / volts, 'part': 'battery'})

        volts_2 = volts_1 - amps * self.speed_controller.resistance
        efficiencies.append({'volts': volts_2, 'amps': amps, 'efficiency': volts_2 / volts_1, 'part': 'esc'})

        volts_3 = volts_2 * self.throttle_efficiency(throttle)
        efficiencies.append({'volts': volts_3, 'amps': amps, 'efficiency': volts_3 / volts_2, 'part': 'throttle'})

        motor_output = self.motor.run(volts_3, amps) # type: ignore
        efficiencies.append({
            'volts': motor_output['volts_out'],
            'amps': motor_output['amps_out'],
            'efficiency': motor_output['efficiency'],
            'part': 'motor',
        })

        torque = motor_output['torque']

        solution_data = {
            'amps': amps,
            'throttle': throttle,
            'rpm': motor_output['rpm'],
            'electrical_power_true': volts * amps,
            'electrical_power': volts_1 * amps,
            'mechanical_power': motor_output['power_out'],
            'torque': torque
        }

        return solution_data, efficiencies

    def _make_solution_function(self, method: str) -> _SolutionStrategy:
        return _SolutionStrategy(method, self)


class _SolutionStrategy:
    def __init__(self, method: str, propsys: PropulsionSystem) -> None:
        self.method: str = method
        self.propsys: PropulsionSystem = propsys
        if method == 'amps':
            self.solver = self.amps
        elif method == 'rpm':
            self.solver = self.rpm
        elif method == 'throttle':
            self.solver = self.throttle
        elif method == 'torque':
            self.solver = self.torque
        elif method == 'electrical_power':
            self.solver = self.electrical_power
        elif method == 'electrical_power_true':
            self.solver = self.electrical_power_true
        elif method == 'mechanical_power':
            self.solver = self.mechanical_power
        else:
            raise AssertionError(f'Invalid parameter {method} chosen for solution')

    def solve(self, value, rho, CP):
        return self.solver(value, rho, CP)

    def _throttle_from_amps(self, amps: float, rho: float, CP: float,
                            throttle_init: float = 1.0, max_iter: int = 10) -> tuple[float, float]:
        V = self.propsys.battery.volts
        I0 = self.propsys.motor.no_load_amps
        kv = self.propsys.motor.kv / 60
        D = self.propsys.propeller.diameter

        def _update_throttle(t):
            e = self.propsys.throttle_efficiency(t)
            R = (self.propsys.battery.resistance *
                 e + self.propsys.speed_controller.resistance *
                 e + self.propsys.motor.resistance)

            new_t = 1 / (V * e) * (R * amps + np.sqrt((amps - I0) / (kv**3 * rho * D**5 * CP)))
            t_prime = (R + np.sqrt((-I0 + amps) / (CP * D ** 5 * kv**3 * rho)) / (2 * (-I0 + amps))) / (V * e)

            return new_t, t_prime

        throttle = throttle_init
        throttle_prime = 0.0
        for _ in range(max_iter):
            new_throttle, throttle_prime = _update_throttle(throttle)
            if abs(new_throttle - throttle) < 1e-4:
                return new_throttle, throttle_prime
            throttle = new_throttle

        return throttle, throttle_prime

    def amps(self, amps: float, rho: float, CP: float) -> tuple[float, float]:
        throttle, _ = self._throttle_from_amps(amps, rho, CP)
        return throttle, amps

    def rpm(self, rpm: float, rho: float, CP: float) -> tuple[float, float]:
        n = rpm / 60
        I0 = self.propsys.motor.no_load_amps
        kv = self.propsys.motor.kv / 60
        D = self.propsys.propeller.diameter

        amps = kv * rho * n ** 2 * D ** 5 * CP + I0

        t = self.amps(amps, rho, CP)[0]

        return t, amps

    def torque(self, torque: float, rho: float, CP: float) -> tuple[float, float]:
        kv = self.propsys.motor.kv / 60
        amps = kv * 2 * np.pi * torque + self.propsys.motor.no_load_amps
        result = self.amps(amps, rho, CP)
        return result

    def throttle(self, throttle: float, rho: float, CP: float) -> tuple[float, float]:
        def _fn(a):
            t, t_prime = self._throttle_from_amps(a, rho, CP, throttle, max_iter=1)
            return (throttle - t) ** 2, -2 * (throttle - t) * t_prime

        e = self.propsys.throttle_efficiency(throttle)
        R = (self.propsys.battery.resistance * e +
             self.propsys.speed_controller.resistance * e +
             self.propsys.motor.resistance)
        I0 = self.propsys.motor.no_load_amps
        V = self.propsys.battery.volts

        min_amps = I0 * 1.1
        max_amps = throttle * V / (e * R)
        max_amps = max_amps
        init_guess = (3 * min_amps + max_amps) / 4
        bounds = [(min_amps, max_amps)]

        amps = float(minimize(_fn, np.array(init_guess), bounds=bounds, method='L-BFGS-B', jac=True).x[0])

        return throttle, amps

    def electrical_power(self, elec_power: float, rho: float, CP: float) -> tuple[float, float]:
        V = self.propsys.battery.volts

        def _fn(a):
            t, t_prime = self._throttle_from_amps(a, rho, CP)
            Rb = self.propsys.battery.resistance
            P = t * V * a - Rb * a ** 2
            P_prime = V * (a * t_prime + t) - 2 * a * Rb
            return (elec_power - P) ** 2, -2 * (elec_power - P) * P_prime

        I0 = self.propsys.motor.no_load_amps
        min_amps = I0 * 1.1
        init_guess = 2 * min_amps
        bounds = [(min_amps, None)]

        amps = float(minimize(_fn, np.array(init_guess), bounds=bounds, method='L-BFGS-B', jac=True).x[0]) # type: ignore
        throttle, _ = self._throttle_from_amps(amps, rho, CP)

        return throttle, amps

    def electrical_power_true(self, elec_power: float, rho: float, CP: float) -> tuple[float, float]:
        V = self.propsys.battery.volts

        def _fn(a):
            t, t_prime = self._throttle_from_amps(a, rho, CP)
            P = t * V * a
            P_prime = V * (a * t_prime + t)
            return (elec_power - P) ** 2, -2 * (elec_power - P) * P_prime

        I0 = self.propsys.motor.no_load_amps

        min_amps = I0 * 1.1
        init_guess = 2 * min_amps
        bounds = [(min_amps, None)]

        amps = float(minimize(_fn, np.array(init_guess), bounds=bounds, method='L-BFGS-B', jac=True).x[0]) # type: ignore
        throttle, _ = self._throttle_from_amps(amps, rho, CP)

        return throttle, amps

    def mechanical_power(self, mech_power: float, rho: float, CP: float) -> tuple[float, float]:
        V = self.propsys.battery.volts
        D = self.propsys.propeller.diameter
        kv = self.propsys.motor.kv / 60

        def _fn(a):
            t, t_prime = self._throttle_from_amps(a, rho, CP)
            e = self.propsys.throttle_efficiency(t)
            e_prime = self.propsys.throttle_efficiency_derivative(t)
            R = (self.propsys.battery.resistance * e +
                 self.propsys.speed_controller.resistance * e +
                 self.propsys.motor.resistance)
            R_prime = (self.propsys.battery.resistance + self.propsys.speed_controller.resistance) * e_prime * t_prime
            P = rho * D ** 5 * CP * kv ** 3 * (t * V * e - R * a) ** 3
            P_prime = rho * D ** 5 * CP * kv ** 3 * 3 * (t * V * e - R * a) ** 2 * (
                    V * t_prime * e + V * e_prime * t - R_prime * a - R)
            return (mech_power - P) ** 2, -2 * (mech_power - P) * P_prime

        I0 = self.propsys.motor.no_load_amps
        min_amps = I0 * 1.1
        init_guess = 2 * min_amps
        bounds = [(min_amps, None)]

        amps = float(minimize(_fn, np.array(init_guess), bounds=bounds, method='L-BFGS-B', jac=True).x[0]) # type: ignore
        throttle, _ = self._throttle_from_amps(amps, rho, CP)

        return throttle, amps

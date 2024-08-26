import alphaplane as ap
import numpy as np
import pickle
import copy
import os

from scipy.optimize import minimize
from scipy.interpolate import interp1d


def make_prop(_x):
    chord_pars = _x[:5]
    twist_pars = _x[5:]
    return ap.Propeller.from_shape_parameters(
        chord_pars, twist_pars, diameter,
    )


def fmt(c):
    if c < 1e1:
        return f'{c:6.4f}'
    elif c < 1e2:
        return f'{c:6.3f}'
    elif c < 1e3:
        return f'{c:6.2f}'
    elif c < 1e4:
        return f'{c:6.1f}'
    elif c < 1e5:
        return f' {int(c):5d}'
    elif c < 1e6:
        return f'{int(c):5d}'
    else:
        return '######'


def cost(_x):
    prop = make_prop(_x)
    positions = np.array([0.25, 0.5, 0.75]) * (prop.radius - prop.stations[0]) + prop.stations[0]
    prop.set_airfoils(list(zip(positions, foils)))
    prop.analysis.analyze_airfoils(
        condition=airfoil_condition,
        model_size='medium'
    )
    gmp = ap.PropulsionSystem(prop, motor, battery, esc)
    score = 0
    rpm = 0
    for v, w in zip(velocities, weights):
        results, _, blade_data = gmp.run(
            velocity=v,
            parameter='electrical_power',
            value=681,
            density=1.08,
            dyn_viscosity=1.81e-5,
            propeller_method='qprop'
        )
        if not blade_data.empty and len(results) > 0:
            reynolds_mean = blade_data["reynolds"].mean()
            alpha_max = blade_data["alpha"].max()
            Cl_min = blade_data["CL"].min()
            Cl_max = blade_data["CL"].max()
            Cl_mean = blade_data["CL"].mean()
            thrust = results['thrust']
            rpm += results['rpm']*w
            score += thrust*w
        else:
            return 0
        
        if blade_data is not None:
            # blade_data.to_csv(f'blade_data_{round(v):02d}.csv')
            pass

    # relevant parameters:
    chord_00 = prop.chords_function(prop.stations[0])
    chord_75 = prop.chords_function(0.75 * prop.radius)
    chord_90 = prop.chords_function(0.90 * prop.radius)

    print(f'score: {fmt(score)} [-]  chord_00: {fmt(100*chord_00)} [cm]  '
          f'chord_75: {fmt(100*chord_75)} [cm]  chord_90: {fmt(100*chord_90)} [cm]  rpm: {fmt(rpm)} [min^-1]  '
          f'pitch75: {fmt(prop.pitch75_inches)} [in]  Cl_min: {fmt(Cl_min)}  '
          f'Cl_max: {fmt(Cl_max)}  Cl_mean = {fmt(Cl_mean)}  alpha_max: {fmt(alpha_max)}  '
          f'Re_mean: {fmt(reynolds_mean)}')
    bests.append(np.maximum(score, bests[-1]))
    return -score


method = 'Powell'
chord_params = 5
twists_params = 5
diameter = 0.58
thickness_function = interp1d(
    [0.06, 0.15, 0.29], [0.09, 0.11, 0.115],
    bounds_error=False,
    fill_value=(0.09, 0.115)  # type: ignore
)
airfoil_condition = ap.AirfoilCondition(
    alpha=np.arange(-10, 20, 1),
    reynolds=[5e4, 7.5e4, 1e5, 1.25e5, 1.5e5,
              1.75e5, 2e5, 3e5, 4e5, 5e5, 6e5, 1e6],
    mach=np.arange(0.0, 0.8, 0.1),
    # xtr_top=0.0,
    # xtr_bottom=0.0,
    n_crit=5,
)
pre_run_condition = ap.AirfoilCondition(
    alpha=np.linspace(-10, 20, 100),
    reynolds=1.5e5,
    mach=0.2,
)
pre_run_condition.fill_with_other(airfoil_condition)

camber = 0.035
thickness = 0.1
gap = 0.025
foil = ap.Airfoil.from_naca('4412')
foil.scale_camber_and_thickness(newcamber=camber, newthickness=thickness)
foil.set_TE_gap(gap)
foil.repanel(201)
foil.repair_geometry()

cambers = [0.0420, 0.0403, 0.0358]
foils: list[ap.Airfoil] = []
for c in cambers:
    current_foil = copy.deepcopy(foil)
    current_foil.scale_camber_and_thickness(newcamber=c)
    foils.append(current_foil)
for _foil in foils:
    _foil.analysis.run(airfoil_condition, method='neuralfoil', model_size='xxxlarge')

motor = ap.ElectricMotor.from_database('4035-250')
battery = ap.Battery.from_lipo(6, volts_per_cell=3.8)
esc = ap.SpeedController(resistance=0.0001)
velocities = np.array([3.0, 6.0, 9.0, 12, 15, 16])
weights = np.array([1, 2, 3, 4, 3, 3])
weights = weights/np.sum(weights)

bests = [0]
init_guess = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
bounds = ([(0.0, 2.0)] + [(-1.0, 1.0)] * 4) * 2
res = minimize(cost, x0=init_guess, bounds=bounds, method=method)
best_prop = make_prop(res.x)
best_prop.show()

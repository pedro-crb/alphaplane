import matplotlib.pyplot as plt
import scipy.optimize as opt
import alphaplane as ap
import numpy as np

from alphaplane.numerical_tools.array_operations import monotonic_indices
from neuralfoil_standalone import get_aero_with_corrections
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def kulfan_from_x(_x):
    camber = _x[:camber_pars]
    thickness = _x[camber_pars:]
    foil = ap.Airfoil.from_cst(camber, thickness, TE_gap=TE_gap)
    return foil.neuralfoil_parameters()


def update_plot(foil):
    x_upper = foil.upper_coords[:, 0]
    y_upper = foil.upper_coords[:, 1]
    x_lower = foil.lower_coords[:, 0]
    y_lower = foil.lower_coords[:, 1]
    line_current.set_xdata(np.concatenate([x_upper, x_lower[::-1]]))
    line_current.set_ydata(np.concatenate([y_upper, y_lower[::-1]]))
    plt.pause(0.00001)


def wiggles(data):
    data = np.asarray(data)
    peaks, _ = find_peaks(data)
    valleys, _ = find_peaks(-data)
    extrema = np.sort(np.concatenate((peaks, valleys, [0, len(data) - 1])))

    stack = []
    counts = []
    for i in range(len(extrema) - 1):
        stack.append(data[extrema[i]])
        while len(stack) >= 3 and abs(stack[-2] - stack[-3]) >= 0.0:
            cycle_range = abs(stack[-2] - stack[-3])
            stack.pop()
            stack.pop()
            counts.append(cycle_range)

    for i in range(len(stack) - 1):
        cycle_range = abs(stack[i] - stack[i + 1])
        counts.append(cycle_range / 2)

    intensity = np.sum(np.array(counts) ** 0.5)
    return intensity


def cost(_x):
    kulfan_params = kulfan_from_x(_x)
    foil = ap.Airfoil.from_neuralfoil(kulfan_params, name='Optimized Airfoil')
    foil_x = foil.upper_coords[:, 0]
    assert condition.alpha is not None and condition.reynolds is not None \
        and condition.mach is not None and condition.n_crit is not None
    results = get_aero_with_corrections(
        kulfan_parameters=kulfan_params,
        alpha=condition.alpha,
        Re=condition.reynolds,
        mach=condition.mach,
        n_crit=condition.n_crit,
        model_size=model_size,
    )

    CL = results['CL']
    CL_indices = monotonic_indices(CL)
    CL = CL[CL_indices]  # type: ignore
    CL_cost = 10 * np.maximum(np.max(CL_targets) - np.max(CL), 0.0)
    CL_cost += 10 * np.maximum(np.min(CL) - np.min(CL_targets), 0.0)

    CD = results['CD'][CL_indices]  # type: ignore
    CD = np.where(CD > 0, CD, -100 * CD)
    CD_fit = interp1d(CL, CD, fill_value=(
        1, 1), bounds_error=False, kind='quadratic')  # type: ignore
    CD_at_targets = CD_fit(CL_targets)
    CD_at_targets = np.where(
        CD_at_targets > 0, CD_at_targets, -100 * CD_at_targets)
    CD_cost = np.dot(CD_at_targets, CL_weights)/np.sum(CL_weights)

    CM = results['CM'][CL_indices]  # type: ignore
    CM_cost = 3 * np.maximum(CM_min - np.min(CM), 0.0)
    # CM_cost = -CM_weight * np.min(CM)

    confidences = results['analysis_confidence'][CL_indices]  # type: ignore
    confidence_fit = interp1d(CL, confidences, fill_value=(
        0, 0), bounds_error=False, kind='quadratic')  # type: ignore
    confidence_cost = np.dot(
        np.maximum(0.0, confidence_min -
                   confidence_fit(CL_targets)), CL_weights
    )/np.sum(CL_weights)

    thickness_fn = interp1d(
        foil_x, foil.upper_coords[:, 1] - foil.lower_coords[:, 1])
    thickness_cost = np.maximum(0.0, -np.min(thickness_fn(foil_x)))
    thickness_cost += np.maximum(20 *
                                 (maxthickness_min - foil.thickness_max), 0)
    thickness_cost += np.maximum(20 *
                                 (foil.thickness_max - maxthickness_max), 0)
    for pos, thick in zip(thickness_check_positions, thickness_check_values):
        thickness_cost += np.maximum(15 * (thick - thickness_fn(pos)), 0)

    camber_cost = np.maximum(20 * (foil.camber_max - maxcamber_max), 0)

    wiggles_upper = wiggles(foil.upper_coords[:, 1])
    wiggles_lower = wiggles(foil.lower_coords[:, 1])
    wiggles_cost = 4*(np.maximum(wiggles_upper - 0.4, 0) +
                      np.maximum(wiggles_lower - 0.4, 0))

    final_cost = CL_cost + CD_cost + CM_cost + confidence_cost + \
        thickness_cost + camber_cost + wiggles_cost
    global iteration
    iteration += 1

    def fmt(c):
        if c < 1e1:
            formatted = f'{c:6.4f}'
        elif c < 1e2:
            formatted = f'{c:6.3f}'
        elif c < 1e3:
            formatted = f'{c:6.2f}'
        elif c < 1e4:
            formatted = f'{c:6.1f}'
        elif c < 1e5:
            formatted = f' {int(c):5d}'
        elif c < 1e6:
            formatted = f'{int(c):5d}'
        else:
            return '######'

        red = 255
        green = int(75 + 180*np.exp(-np.minimum(np.abs(30 * c), 100.0)))
        blue = int(65 + 190*np.exp(-np.minimum(np.abs(50 * c), 100.0)))
        color_code = f'\033[38;2;{red};{green};{blue}m'
        reset_code = '\033[0m'
        return f'{color_code}{formatted}{reset_code}'

    print(f'i -- \033[38;2;255;255;255m{iteration:5d}\033[0m   costs -- CL: {fmt(CL_cost)} | CD: {fmt(CD_cost)} | '
          f'CM: {fmt(CM_cost)} | confidence: {fmt(confidence_cost)} | '
          f'thick: {fmt(thickness_cost)} | camber: {fmt(camber_cost)} | '
          f'wiggles: {fmt(wiggles_cost)} | total: {fmt(final_cost)}')
    update_plot(foil)
    return final_cost

if __name__ == '__main__':
    # setup
    np.random.seed(seed=789)
    method = 'Nelder-Mead'
    camber_pars = 5
    thickness_pars = 5
    model_size = 'xlarge'
    CL_targets = np.array([0.2, 0.7, 1.0, 1.1, 1.3, 1.4, 1.5, 1.6, 1.7])
    CL_weights = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 3.0, 1.0])
    CM_min = -0.2
    confidence_min = 0.90
    maxthickness_min = 0.10
    maxthickness_max = 0.15
    maxcamber_max = 0.1
    thickness_check_positions = np.array([0.8, 0.9])
    thickness_check_values = np.array([0.020, 0.015])
    TE_gap = 0.03
    condition = ap.AirfoilCondition(
        alpha=np.linspace(-20, 20, 61),
        reynolds=1.0e5,
        mach=0.1,
        n_crit=5,
        # xtr_top=0.0,
        # xtr_bottom=0.0
    )

    # initial guess
    init_airfoil = ap.Airfoil.from_database('e63')
    init_airfoil.scale_camber_and_thickness(newcamber=0.037, newthickness=0.1)
    init_airfoil.set_TE_gap(TE_gap)
    cst_initial = init_airfoil.cst_parameters(camber_pars, thickness_pars)

    # benchmarks
    benchmark_airfoils = [
        ap.Airfoil.from_naca('2412'),
    ]
    for foil in benchmark_airfoils:
        foil.set_TE_gap(TE_gap)
        foil.scale_camber_and_thickness(newcamber=0.037, newthickness=0.1)
        foil.repair_geometry()

    # Prepare plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    previous_airfoils = []
    line_current, = ax.plot(*init_airfoil.coords.T, 'k-',
                            label='Current Airfoil', zorder=11)
    line_initial, = ax.plot(*init_airfoil.coords.T, 'b--',
                            label='Initial Airfoil', zorder=10)
    ax.legend()
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(which='both')
    ax.minorticks_on()
    ax.grid(which='minor', linestyle='--', linewidth=0.5, color='#cccccc')
    plt.tight_layout()
    plt.draw()

    # global counter
    iteration = 0

    # bounds
    x0 = np.array(list(np.array(cst_initial[0])) + list(np.array(cst_initial[1])))
    bounds_camber = [
        (-0.3, 0.9),
        (-0.4, 0.8),
        (-0.3, 0.8),
        (-0.4, 0.5),
        (-0.3, 0.9),
    ]
    bounds_thickness = [
        (-0.0, 1.0),
        (-0.3, 1.2),
        (-0.3, 1.1),
        (-0.3, 1.0),
        (-0.3, 1.1),
    ]
    bounds = bounds_camber[:camber_pars] + bounds_thickness[:thickness_pars]

    # solution
    initial_simplex = (
        (0.5 + 1 * np.random.random((len(x0) + 1, len(x0))))
        * x0
    )
    initial_simplex[0, :] = x0
    res = opt.minimize(
        fun=cost,
        x0=x0,
        method=method,
        options={
            'maxiter': 2 * 10 ** 3,
            'initial_simplex': initial_simplex,
            'xatol': 1e-3,
            'fatol': 1e-6,
            'adaptive': False,
        }
    )

    # final plots
    plt.close(fig)
    plt.ioff()
    final_foil = ap.Airfoil.from_neuralfoil(kulfan_from_x(res.x))
    final_foil.repair_geometry()
    final_foil.smooth_airfoil()
    final_foil.plot()
    init_airfoil.analysis.run(condition)
    for _foil in benchmark_airfoils:
        _foil.smooth_airfoil()
        _foil.analysis.run(condition)
    final_foil.analysis.run(condition)
    # final_foil.to_dat('optimized_foil.dat')
    ap.Airfoil.AirfoilAnalysis.plot_multiple(
        [init_airfoil, final_foil] + benchmark_airfoils, condition=condition)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib
import os
from datetime import datetime
import json
import sys
import os

# Add current directory to sys.path so we can import plotting_styles
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from plotting_styles import PStyle, apply_ticks, apply_spines
except ImportError:
    # If plotting_styles is not found, provide a simple fallback
    class PStyle:
        def __init__(self):
            self.pred_lw = 2.0

    def apply_ticks(ax):
        pass

    def apply_spines(ax):
        pass

    print("Warning: plotting_styles module not found. Using default styles.")

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def ode_system(t, y):
    """
    Define the ODE system (consistent with the user-provided equations).

    Mapping:
        y[0] = x1  <-> x(1)
        y[1] = x2  <-> x(2)
    """
    x1, x2 = y

    # dx(1)/dt
    dx1_dt = (
        14.4129 * x2
        - 1.10396 * x1 * x2
        - 0.706969 * x2 * x2
        + 0.511632 * x1 * x2 * x2
        - 0.274129 * x1 * x1
        - 0.204207 * x1 * x1 * x1
        + 0.186406 * x1 * x1 * x1 * x2
        + 0.111268 * x2 * x2 * x2 * x2
        + 0.110337 * x2 * x2 * x2
        + 0.0996347 * x1 * x2 * x2 * x2
        + 0.0970819 * x1
        + 0.0234364 * x1 * x1 * x1 * x1
    )

    # dx(2)/dt
    dx2_dt = (
        -17.5155 * x1
        - 1.42427 * x2 * x2
        - 1.4151 * x2
        + 1.13042 * x1 * x2
        + 1.03413 * x1 * x1
        + 0.98089 * x1 * x2 * x2
        + 0.662696 * x1 * x1 * x1
        - 0.461067 * x1 * x1 * x1 * x2
        - 0.417805 * x1 * x1 * x2
        + 0.412774 * x2 * x2 * x2
        - 0.409313 * x1 * x1 * x1 * x1
        + 0.33917 * x2 * x2 * x2 * x2
    )

    return [dx1_dt, dx2_dt]


def check_oscillation_decay(sol, threshold=0.01):
    """
    Check whether the solution is oscillatory and whether the oscillation decays.

    Returns:
        (is_oscillating, is_decaying, oscillation_amplitude)
    """
    if not sol.success or len(sol.y[0]) < 100:
        return False, False, 0

    x1 = sol.y[0]
    x2 = sol.y[1]

    # Use standard deviation as a proxy for oscillation amplitude/strength
    x1_std = np.std(x1)
    x2_std = np.std(x2)
    oscillation_amplitude = max(x1_std, x2_std)

    # Oscillatory if amplitude exceeds threshold
    is_oscillating = oscillation_amplitude > threshold

    # Decay check: std in the last quarter is smaller than the first quarter
    n = len(x1)
    first_quarter = x1[:n // 4]
    last_quarter = x1[-n // 4:]

    first_std = np.std(first_quarter)
    last_std = np.std(last_quarter)

    # Consider "decaying" if it shrinks by at least 50%
    is_decaying = last_std < first_std * 0.5

    return is_oscillating, is_decaying, oscillation_amplitude


def find_stable_initial_conditions(t_span, t_eval, search_range=(-5, 5), num_points=20):
    """
    Search for "stable" initial conditions (i.e., those that do not show oscillation decay).
    """
    print("Searching for stable initial conditions...")

    stable_conditions = []
    x1_range = np.linspace(search_range[0], search_range[1], num_points)
    x2_range = np.linspace(search_range[0], search_range[1], num_points)

    for x1_init in x1_range:
        for x2_init in x2_range:
            y0 = [x1_init, x2_init]
            try:
                sol = solve_ivp(
                    ode_system, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8
                )

                if sol.success:
                    is_oscillating, is_decaying, amplitude = check_oscillation_decay(sol)

                    # Keep conditions that are NOT oscillation-decaying
                    if (not is_oscillating) or (not is_decaying):
                        stable_conditions.append({
                            'initial': y0,
                            'amplitude': amplitude,
                            'is_oscillating': is_oscillating,
                            'is_decaying': is_decaying
                        })
                        print(
                            f"Found stable condition: {y0}, "
                            f"oscillating: {is_oscillating}, decaying: {is_decaying}, "
                            f"amplitude: {amplitude:.4f}"
                        )

            except Exception:
                continue

    return stable_conditions


def load_initial_conditions_from_file(file_path="sample_initial_conditions.txt"):
    """
    Load initial conditions from a file.

    Expected format:
        Each non-empty, non-comment line contains: x1,x2
    """
    print(f"\n=== Loading initial conditions from file: {file_path} ===")

    initial_conditions = []

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        x1, x2 = map(float, line.split(','))
                        initial_conditions.append([x1, x2])
                        print(f"  Loaded initial condition: [{x1}, {x2}]")
                    except ValueError:
                        print(f"  Skipping invalid line {line_num}: {line}")

        print(f"Successfully loaded {len(initial_conditions)} initial conditions from file.")

    except FileNotFoundError:
        print(f"File {file_path} not found. Using default initial conditions.")
        initial_conditions = [
            [2.6633165829145735, 1.0552763819095485],
            [1.9872837,  1.2634588],
            [2.0429988, -0.11742538],
            [1.0, 0.0],
            [0.0, 1.0]
        ]
        for i, cond in enumerate(initial_conditions, 1):
            print(f"  Default initial condition {i}: {cond}")

    return initial_conditions


def hide_all_axes(ax, keep_ticks=True):
    """
    Remove axis frame and tick labels.

    Args:
        keep_ticks (bool):
            - True: keep tick marks (small tick lines) but remove tick labels.
            - False: remove tick marks as well (fully clean).
    """
    # 1) Hide all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 2) Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 3) Keep or remove tick marks
    if keep_ticks:
        ax.tick_params(
            axis="both",
            which="both",
            length=6,   # You can set to 0 to remove tick marks
            width=1.5,
            direction="out"
        )
    else:
        ax.tick_params(
            axis="both",
            which="both",
            length=0
        )


def create_output_folder():
    """
    Create a new output subfolder under plots_1/ with a timestamp.
    """
    plots_dir = "plots_1"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Create a timestamped run folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(plots_dir, f"user_input_run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    return run_folder


def plot_ode_timeseries(initial_file="sample_initial_conditions.txt", search_stable=False):
    """
    Plot time-series curves of the ODE system (initial conditions loaded from file).

    Args:
        initial_file (str): file name containing initial conditions
        search_stable (bool): whether to search for additional stable initial conditions
    """
    # Create output folder
    output_folder = create_output_folder()
    print(f"Figures will be saved to: {output_folder}")

    # Load initial conditions from file
    initial_conditions = load_initial_conditions_from_file(initial_file)

    # Optionally search for stable initial conditions
    if search_stable:
        print("\nSearching for stable initial conditions...")
        t_span = (0, 3.075)
        t_eval = np.linspace(0, 3.075, 1000)
        stable_conditions = find_stable_initial_conditions(t_span, t_eval)

        if stable_conditions:
            print(f"\nFound {len(stable_conditions)} stable initial conditions.")
            # Automatically append the first 3 stable conditions
            for cond in stable_conditions[:3]:
                initial_conditions.append(cond['initial'])
                print(f"Added stable condition: {cond['initial']}")
        else:
            print("No stable initial conditions found.")

    # Time window [0, 3.075]
    t_span = (0, 3.075)
    t_eval = np.linspace(0, 3.075, 1000)

    # Create figure (2 rows: x1(t) and x2(t))
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Color scheme: first three are colored, remaining are light gray
    style = PStyle()
    colors = []
    for i in range(len(initial_conditions)):
        if i == 0:
            colors.append('#d06569')  # First IC: red
        elif i == 1:
            colors.append('#4FA8D5')  # Second IC: blue
        elif i == 2:
            colors.append('#ECBC91')  # Third IC: orange
        else:
            colors.append('#D3D3D3')  # Others: light gray

    results = []

    for i, (y0, color) in enumerate(zip(initial_conditions, colors)):
        try:
            # Solve ODE (try tighter tolerances first)
            sol = solve_ivp(
                ode_system, t_span, y0, t_eval=t_eval,
                method='RK45', rtol=1e-6, atol=1e-8
            )

            # If failed, relax tolerances
            if not sol.success:
                print("  High-precision solve failed; trying looser tolerances...")
                sol = solve_ivp(
                    ode_system, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-4, atol=1e-6
                )

            if sol.success:
                # Analyze oscillation/decay behavior
                is_oscillating, is_decaying, amplitude = check_oscillation_decay(sol)

                # Plot x1(t)
                axes[0].plot(sol.t, sol.y[0], color=color, linewidth=style.pred_lw)

                # Plot x2(t)
                axes[1].plot(sol.t, sol.y[1], color=color, linewidth=style.pred_lw)

                # Record analysis
                results.append({
                    'initial': y0,
                    'is_oscillating': bool(is_oscillating),
                    'is_decaying': bool(is_decaying),
                    'amplitude': float(amplitude)
                })

                print(
                    f"IC {i + 1} ({y0[0]:.3f}, {y0[1]:.3f}): "
                    f"oscillating={is_oscillating}, decaying={is_decaying}, "
                    f"amplitude={amplitude:.4f}"
                )
            else:
                print(f"Solve failed for initial condition {y0}: {sol.message}")

        except Exception as e:
            print(f"Error solving for initial condition {y0}: {str(e)}")

    # Apply style (keep only ticks/spines as defined in plotting_styles)
    apply_ticks(axes[0])
    apply_ticks(axes[1])
    apply_spines(axes[0])
    apply_spines(axes[1])

    plt.tight_layout()

    # Save time-series figure (PNG + SVG)
    timeseries_png_path = os.path.join(output_folder, "timeseries.png")
    timeseries_svg_path = os.path.join(output_folder, "timeseries.svg")

    plt.savefig(timeseries_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(timeseries_svg_path, dpi=300, bbox_inches='tight')
    print(f"Time-series figure saved: {timeseries_png_path}")
    print(f"Time-series figure saved: {timeseries_svg_path}")
    plt.close()  # Close to release memory

    # Also plot the phase plane
    plt.figure(figsize=(10, 8))
    for i, (y0, color) in enumerate(zip(initial_conditions, colors)):
        try:
            sol = solve_ivp(
                ode_system, t_span, y0, t_eval=t_eval,
                method='RK45', rtol=1e-6, atol=1e-8
            )

            if not sol.success:
                sol = solve_ivp(
                    ode_system, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-4, atol=1e-6
                )

            if sol.success:
                # Use a slightly thinner line in phase space to reduce clutter
                plt.plot(sol.y[0], sol.y[1], color=color, linewidth=3)
                # Mark the initial point
                plt.plot(sol.y[0][0], sol.y[1][0], 'o', color=color, markersize=8)

        except Exception as e:
            print(f"Error plotting phase plane for initial condition {y0}: {str(e)}")

    # Axis settings: equal aspect ratio + hide all axes/ticks if desired
    plt.axis('equal')

    # Apply axis cleanup: remove everything including ticks
    hide_all_axes(plt.gca(), keep_ticks=False)

    plt.tight_layout()

    # Save phase plane figure (PNG + SVG)
    phase_png_path = os.path.join(output_folder, "phase_plane.png")
    phase_svg_path = os.path.join(output_folder, "phase_plane.svg")

    plt.savefig(phase_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(phase_svg_path, dpi=300, bbox_inches='tight')
    print(f"Phase-plane figure saved: {phase_png_path}")
    print(f"Phase-plane figure saved: {phase_svg_path}")
    plt.close()  # Close to release memory

    # Save analysis results to JSON
    results_path = os.path.join(output_folder, "analysis_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Analysis results saved: {results_path}")

    return output_folder, results


if __name__ == "__main__":
    # Configuration: modify file name and options here
    INITIAL_FILE = "initial.txt"   # Initial conditions file name
    SEARCH_STABLE = False          # Whether to search for stable initial conditions

    print("Starting time-series plotting for the ODE system...")
    print("Integration time window: [0, 3.075]")
    print("Equation source: (Updated by User)")
    print(f"Initial condition file: {INITIAL_FILE}")
    print(f"Search stable conditions: {SEARCH_STABLE}")
    print()

    output_folder, results = plot_ode_timeseries(INITIAL_FILE, SEARCH_STABLE)

    print(f"\nDone. All files saved to: {output_folder}")
    print("\nAnalysis summary:")
    for i, result in enumerate(results):
        initial_str = f"({result['initial'][0]:.3f}, {result['initial'][1]:.3f})"
        print(
            f"  IC {i + 1} {initial_str}: "
            f"oscillating={result['is_oscillating']}, "
            f"decaying={result['is_decaying']}, "
            f"amplitude={result['amplitude']:.4f}"
        )

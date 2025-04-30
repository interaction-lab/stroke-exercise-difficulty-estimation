import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from econml.dml import CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter

from src.utils import get_train_test_data, get_features
from src.viz.plot_utils import get_groups, get_X



# --- Constants ---
DEFAULT_SEED = 169
DEFAULT_MAX_DEPTH = 3
DEFAULT_MIN_SAMPLES_LEAF_INTERP = 0.02
DEFAULT_N_ESTIMATORS_CF = 600
DEFAULT_MIN_SAMPLES_LEAF_CF = 10
DEFAULT_N_ESTIMATORS_GBR = 50
DEFAULT_LEARNING_RATE_GBR = 0.01
PLOT_ALPHA_POINTS = 0.15
PLOT_SIZE_POINTS = 10
PLOT_ALPHA_GT = 0.7
PLOT_SIZE_GT = 70
PLOT_CMAP = 'viridis'
PLOT_VIEW_INIT = (42, -70)
PLOT_X_LIMIT = (-.3, .3)
PLOT_X_TICKS = [-.3, -.2, -.1, 0, .1, .2, .3]
PLOT_X_TICKLABELS = ['30','20','10','0','10','20','30']
PLOT_Y_LIMIT = (0, .3)
PLOT_Y_TICKS = [0, .1, .2, .3]
PLOT_Y_TICKLABELS = ['','10','20','30']
PLOT_Z_LIMIT = (0, .4)
PLOT_Z_TICKS = [0, .1, .2, .3, .4]
PLOT_Z_TICKLABELS = ['','10','20','30', '40']
PLOT_OUTLINE_COLOR = '#FF91AF'
PLOT_HOME_CYLINDER_COLOR = '#ffbf0033'
PLOT_HOME_CYLINDER_RADIUS_INNER = 0.1
PLOT_HOME_CYLINDER_RADIUS_OUTER = 0.3
PLOT_HOME_CYLINDER_HEIGHT = 0.4
PLOT_COLORBAR_SHRINK = 0.7
PLOT_COLORBAR_PAD = 0.05
PLOT_COLORBAR_LABEL = 'Additional Time to Reach (s)'


def setup_3d_plot(ax: plt.Axes):
    """Sets up the common 3D axes limits, labels, and outline."""
    ax.set_box_aspect([2, 1, 1.33])

    # Plot outline and home area cylinders
    outline_color = PLOT_OUTLINE_COLOR
    ax.plot3D([.10, .10], [0,0], [0,PLOT_HOME_CYLINDER_HEIGHT], color=outline_color)
    ax.plot3D([-.10, -.10], [0,0], [0,PLOT_HOME_CYLINDER_HEIGHT], color=outline_color)
    ax.plot3D([.30, .30], [0,0], [0,PLOT_HOME_CYLINDER_HEIGHT], color=outline_color)
    ax.plot3D([-.30, -.30], [0,0], [0,PLOT_HOME_CYLINDER_HEIGHT], color=outline_color)
    ax.plot3D([-.30, -.10], [0,0], [PLOT_HOME_CYLINDER_HEIGHT, PLOT_HOME_CYLINDER_HEIGHT], color=outline_color)
    ax.plot3D([-.30, -.10], [0,0], [.0,.0], color=outline_color)
    ax.plot3D([.10, .30], [0,0], [PLOT_HOME_CYLINDER_HEIGHT, PLOT_HOME_CYLINDER_HEIGHT], color=outline_color)
    ax.plot3D([.10, .30], [0,0], [.0,.0], color=outline_color)
    theta = np.linspace(0, np.pi, 100)
    ax.plot3D(PLOT_HOME_CYLINDER_RADIUS_INNER * np.cos(theta), PLOT_HOME_CYLINDER_RADIUS_INNER * np.sin(theta), [0]*100, color=outline_color)
    ax.plot3D(PLOT_HOME_CYLINDER_RADIUS_INNER * np.cos(theta), PLOT_HOME_CYLINDER_RADIUS_INNER * np.sin(theta), [PLOT_HOME_CYLINDER_HEIGHT]*100, color=outline_color)
    ax.plot3D(PLOT_HOME_CYLINDER_RADIUS_OUTER * np.cos(theta), PLOT_HOME_CYLINDER_RADIUS_OUTER * np.sin(theta), [0]*100, color=PLOT_HOME_CYLINDER_COLOR)
    ax.plot3D(PLOT_HOME_CYLINDER_RADIUS_OUTER * np.cos(theta), PLOT_HOME_CYLINDER_RADIUS_OUTER * np.sin(theta), [PLOT_HOME_CYLINDER_HEIGHT]*100, color=outline_color)

    ax.view_init(*PLOT_VIEW_INIT)
    ax.set_xlim(*PLOT_X_LIMIT)
    ax.set_xticks(PLOT_X_TICKS)
    ax.set_xticklabels(PLOT_X_TICKLABELS, ha='right')
    ax.set_xlabel('Distance from Home Position (cm)', labelpad=15)

    ax.set_ylim(*PLOT_Y_LIMIT)
    ax.set_yticks(PLOT_Y_TICKS)
    ax.set_yticklabels(PLOT_Y_TICKLABELS, ha='left')

    ax.set_zlim(*PLOT_Z_LIMIT)
    ax.set_zticks(PLOT_Z_TICKS)
    ax.set_zticklabels(PLOT_Z_TICKLABELS)
    ax.set_zlabel('Height (cm)', labelpad=15)


def plot_tree(interp: SingleTreeCateInterpreter, pid: int):
    """
    Plots the CATE interpreter tree nodes in 3D space based on feature splits.

    Args:
        interp: The trained SingleTreeCateInterpreter instance.
        pid: The participant ID.
    """
    print(f"Plotting interpreter tree for PID {pid}...")
    X, cols = get_X()
    if X.size == 0 or not cols:
        print("Warning: No data or columns to plot tree.")
        return

    group_dict = get_groups(interp.tree_model_.tree_, cols)
    data = pd.DataFrame(X, columns=cols)
    data['causal_effect'] = 0.0

    # Fill in the causal effects in terms of seconds
    for node, criteria in group_dict.items():
        query = ' and '.join(criteria)
        # print(f"Node {node} query: {query}") # Uncomment for debugging
        try:
            # Ensure the query is valid for the dataframe columns
            data.loc[data.query(query).index, 'causal_effect'] = interp.tree_model_.tree_.value[node][0, 0]
        except Exception as e:
            print(f"Warning: Could not apply query '{query}' for node {node}: {e}")
            # Optionally, skip this node or assign a default value

    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes(projection='3d')
    setup_3d_plot(ax)

    scatter = ax.scatter3D(xs=data['x'], ys=data['y'], zs=data['z'],
                           c=data['causal_effect'], alpha=.45, cmap=PLOT_CMAP)

    fig.tight_layout()

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, location='left', shrink=PLOT_COLORBAR_SHRINK, pad=PLOT_COLORBAR_PAD)
    cbar.set_label(PLOT_COLORBAR_LABEL)
    cbar.solids.set(alpha=1)

    output_dir = f'images/{pid}'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/tree_partitioned_space_{pid}.png')
    if args.verbose:
        plt.show()
    print(f"Interpreter tree plot saved to {output_dir}/tree_partitioned_space_{pid}.png")


def plot_points(Xs: np.ndarray, ys: np.ndarray, pid: int, name: str, s: int = PLOT_SIZE_POINTS,
                alpha: float = PLOT_ALPHA_POINTS, vmin: float = None, vmax: float = None):
    """
    Plots points in 3D space, colored by their corresponding values.

    Args:
        Xs: A numpy array of shape (n_samples, 3) containing the 3D coordinates (x, y, z).
        ys: A numpy array of shape (n_samples,) containing the values to color the points by.
        pid: The participant ID.
        name: The base name for the output file (e.g., 'causal_effect', 'baseline', 'gt').
        s: The size of the markers.
        alpha: The alpha transparency value for the markers.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
    """
    print(f"Plotting {name} for PID {pid}...")
    if Xs.shape[0] == 0:
        print(f"Warning: No data points to plot for {name}.")
        return

    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes(projection='3d')
    setup_3d_plot(ax)

    scatter = ax.scatter3D(xs=Xs[:, 0], ys=Xs[:, 1], zs=Xs[:, 2],
                           c=ys, alpha=alpha, s=s, cmap=PLOT_CMAP, vmin=vmin, vmax=vmax)

    fig.tight_layout()

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax, location='left', shrink=PLOT_COLORBAR_SHRINK, pad=PLOT_COLORBAR_PAD)
    cbar.set_label(PLOT_COLORBAR_LABEL)
    cbar.solids.set(alpha=1)

    output_dir = f'images/{pid}'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{name}_{pid}.png')
    if args.verbose:
        plt.show()
    print(f"{name} plot saved to {output_dir}/{name}_{pid}.png")


def process_participant(pid: int, seed: int, vmin: float = None, vmax: float = None):
    """
    Processes data for a single participant, trains models, and generates plots.

    Args:
        pid: The participant ID to process.
        seed: The random seed for reproducibility.
        vmin: Minimum value for color scaling in plots.
        vmax: Maximum value for color scaling in plots.
    """
    print(f"Processing data for Participant ID: {pid}")

    datasets = get_train_test_data(seed)
    np.random.seed(seed)

    participant_dataset = None
    for dataset in datasets:
        if dataset.get('pid') == pid:
            participant_dataset = dataset
            break

    if participant_dataset is None:
        print(f"Error: Data not found for Participant ID {pid}. Skipping.")
        return

    output_dir = f'images/{pid}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    # STEP 1: Train the causal forest and the interpreter
    print("Step 1: Training Causal Forest and Interpreter...")
    try:
        model = CausalForestDML(
            n_estimators=DEFAULT_N_ESTIMATORS_CF,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF_CF,
            discrete_treatment=True,
            random_state=seed
        )
        X_train = np.concatenate([participant_dataset['stroke_X_train'], participant_dataset['neuro_X_train']])
        y_train = np.concatenate([participant_dataset['stroke_y_train'], participant_dataset['neuro_y_train']])
        T_train = np.zeros(X_train.shape[0])
        T_train[:len(participant_dataset['stroke_X_train'])] = 1

        model.fit(Y=y_train, T=T_train, X=X_train)
        print("Causal Forest trained.")

        interp = SingleTreeCateInterpreter(
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF_INTERP,
            random_state=seed
        )
        interp.interpret(model, X_train)
        print("Interpreter trained.")

        X_plot, _ = get_X()
        if X_plot.size > 0:
            y_causal_effect = model.effect(X_plot)
            plot_tree(interp, pid)
            plot_points(X_plot, y_causal_effect, pid, name='causal_effect', s=PLOT_SIZE_POINTS, alpha=PLOT_ALPHA_POINTS, vmin=vmin, vmax=vmax)
        else:
            print("Warning: No plotting data from get_X(). Skipping causal effect plots.")

    except Exception as e:
        print(f"Error during Step 1: {e}")

    # STEP 2: Plot the baseline
    print("\nStep 2: Training Baseline Models and Plotting...")
    try:
        stroke_model = GradientBoostingRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS_GBR,
            learning_rate=DEFAULT_LEARNING_RATE_GBR,
            max_depth=DEFAULT_MAX_DEPTH,
            random_state=seed
        )
        stroke_model.fit(participant_dataset['stroke_X_train'], participant_dataset['stroke_y_train'])
        print("Stroke baseline model trained.")

        neuro_model = GradientBoostingRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS_GBR,
            learning_rate=DEFAULT_LEARNING_RATE_GBR,
            max_depth=DEFAULT_MAX_DEPTH,
            random_state=seed
        )
        neuro_model.fit(participant_dataset['neuro_X_train'], participant_dataset['neuro_y_train'])
        print("Neuro baseline model trained.")

        X_plot, _ = get_X()
        if X_plot.size > 0:
            y_baseline = stroke_model.predict(X_plot) - neuro_model.predict(X_plot)
            # Using a fixed vmax for baseline plot as per original code
            plot_points(X_plot, y_baseline, pid, name='baseline', s=PLOT_SIZE_POINTS, alpha=PLOT_ALPHA_POINTS, vmin=vmin, vmax=.35)
        else:
             print("Warning: No plotting data from get_X(). Skipping baseline plot.")

    except Exception as e:
        print(f"Error during Step 2: {e}")

    # STEP 3: Plot the ground truth
    print("\nStep 3: Plotting Ground Truth...")
    try:
        # Use stroke_X_train and stroke_y_train_plotting for GT plot
        if participant_dataset['stroke_X_train'].shape[0] > 0:
             plot_points(
                participant_dataset['stroke_X_train'],
                participant_dataset.get('stroke_y_train_plotting', participant_dataset['stroke_y_train']), # Fallback to stroke_y_train
                pid,
                name='ground_truth',
                s=PLOT_SIZE_GT,
                alpha=PLOT_ALPHA_GT,
                vmin=vmin,
                vmax=vmax # Use global vmax for consistency with causal effect
            )
        else:
            print("Warning: No stroke training data available for ground truth plot.")

    except Exception as e:
        print(f"Error during Step 3: {e}")

    print(f"Finished processing for Participant ID: {pid}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate causal effect, baseline, and ground truth plots for a specific participant ID.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose output.'
    )
    parser.add_argument(
        '--pid',
        type=int,
        required=True, # Make PID a required argument
        help='Participant ID to process.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help=f'Random seed for reproducibility (default: {DEFAULT_SEED}).'
    )
    parser.add_argument(
        '--vmin',
        type=float,
        default=0, # Default vmin as per original code
        help='Minimum value for color scale (default: 0).'
    )
    parser.add_argument(
        '--vmax',
        type=float,
        help='Maximum value for color scale (optional). If not provided, inferred by matplotlib.'
    )

    args = parser.parse_args()

    # Validate PID (basic check)
    if args.pid <= 0:
        print("Error: Participant ID must be a positive integer.")
        exit(1) # Exit with a non-zero status code to indicate an error

    # Run the processing function for the specified participant
    process_participant(args.pid, args.seed, args.vmin, args.vmax)
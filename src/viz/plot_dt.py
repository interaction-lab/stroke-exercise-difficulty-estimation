import argparse
import os
import numpy as np

# Assuming these modules are correctly installed and accessible
from src.utils import get_train_test_data
from src.viz.plot_utils import get_groups, get_X # get_groups is imported but not used, kept for completeness if needed elsewhere
from econml.dml import CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter
from graphviz import Source


def plot_decision_tree(dataset: dict, pid: int, seed: int):
    """
    Trains a Causal Forest model and a SingleTreeCateInterpreter,
    then exports and renders the decision tree for a given dataset.

    Args:
        dataset: A dictionary containing the participant's training data
                 ('stroke_X_train', 'stroke_y_train', 'neuro_X_train',
                 'neuro_y_train') and 'pid'.
        pid: The participant ID.
        seed: The random seed for reproducibility.
    """
    print(f"\nProcessing participant PID: {pid}")

    # STEP 1: Prepare data and train the causal forest
    # Concatenate stroke and neuro training data
    X = np.concatenate([dataset['stroke_X_train'], dataset['neuro_X_train']])
    y = np.concatenate([dataset['stroke_y_train'], dataset['neuro_y_train']])

    # Create treatment indicator (1 for stroke data, 0 for neuro data)
    T = np.zeros(X.shape[0], dtype=int)
    T[:len(dataset['stroke_X_train'])] = 1

    print("Training Causal Forest model...")
    model = CausalForestDML(
        n_estimators=600,
        min_samples_leaf=10,
        discrete_treatment=True,
        random_state=seed
    )
    # Fit the model using the prepared data
    model.fit(Y=y, T=T, X=X)
    print("Causal Forest model trained.")

    # STEP 2: Train the interpreter
    print("Training Single Tree CATE Interpreter...")
    interp = SingleTreeCateInterpreter(
        max_depth=3,
        min_samples_leaf=0.02, # min_samples_leaf can be an absolute number or a fraction
        random_state=seed
    )
    # Interpret the trained causal forest model
    interp.interpret(model, X)
    print("Interpreter trained.")

    # STEP 3: Export and render the tree
    output_dir = f'images/{pid}'
    # Ensure directory exists before saving files
    os.makedirs(output_dir, exist_ok=True)

    dot_file_path = os.path.join(output_dir, f'tree_{pid}.dot')
    png_file_path = os.path.join(output_dir, f'tree_{pid}')

    print(f"Exporting tree to {dot_file_path}")
    # Get feature names using the imported get_X function
    _, feature_names = get_X()
    interp.export_graphviz(out_file=dot_file_path, feature_names=feature_names)
    print("Tree exported successfully.")

    print(f"Rendering tree to {png_file_path}.png")
    try:
        # Render the dot file to a PNG image
        Source.from_file(dot_file_path).render(png_file_path, format='png', cleanup=True)
        print("Tree rendered successfully.")
    except Exception as e:
        print(f"Error rendering tree using Graphviz: {e}")
        print("Please ensure Graphviz is installed and in your system's PATH.")
        print("You can install Graphviz from https://graphviz.org/download/")


def main():
    """
    Main function to parse arguments, load data, and plot decision trees
    for specified participant.
    """
    parser = argparse.ArgumentParser(
        description="Train a Causal Forest and plot the decision tree interpreter."
    )
    parser.add_argument(
        '--pid',
        type=int,
        required=True,
        help="Participant ID to process."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=169,
        help='Random seed for reproducibility.'
    )

    # Process command line arguments and set up 
    args = parser.parse_args()
    np.random.seed(args.seed)
    datasets = get_train_test_data(args.seed)

    # Select dataset for the specified participant
    for dataset in datasets:
        current_pid = dataset['pid']

        # Check if a specific PID was requested and if it matches the current dataset's PID
        if args.pid is not None and current_pid != args.pid:
            continue

        # Create the directory for output images if it doesn't exist
        output_dir = f'images/{current_pid}'
        os.makedirs(output_dir, exist_ok=True)

        # Create the decision tree for the current dataset
        plot_decision_tree(dataset, current_pid, args.seed)

    print("\nProcessing complete.")


if __name__ == '__main__':
    main()
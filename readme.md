# Functional Difficulty Estimation

This repository contains the code and resources for the paper "Modeling Personalized Difficulty of Rehabilitation Exercises Using Causal Trees" published at _2025 IEEE/RAS International Conference on Rehabilitation Robotics (ICORR)_. This project contains the code to create personalized difficulty models for rehabilitation tasks, enabling computational adaptation of task difficulty.

<table style="margin-left: auto; margin-right: auto; width: 700px;"> <tr>
    <td style="text-align: center; padding: 10px;">
      <img src="assets/PID27.png" width="300"/>
      <br>PID 27, right-affected
    </td>
    <td style="text-align: center; padding: 10px;">
      <img src="assets/PID31.png" width="300"/>
      <br>PID 31, left-affected
    </td>
  </tr>
</table>

## Modeling Functional Difficulty

Drawing upon the principles of the [Challenge Point Framework](https://en.wikipedia.org/wiki/Challenge_point_framework) (CPF), which posits that motor learning is optimized at an ideal level of task difficulty, this project focuses on the quantitative estimation of **functional task difficulty**. Functional difficulty is defined as the difficulty of a task relative to an individual's specific skill level, a crucial distinction from the inherent or average task difficulty, often referred to as **nominal task difficulty**. Accurately quantifying this personalized difficulty is challenging, largely due to the high variance typically observed in human performance measurements, especially within diverse populations.

To address the challenge of high performance variance and estimate individual-specific task difficulty, we adapt techniques from the field of **heterogeneous treatment effect estimation**. We formalize the concept of individual exercise difficulty for a specific task represented by feature vector $x$ using the **potential outcomes framework**:

$$ \tau(x) = E[Y(1) - Y(0) \mid X=x] $$

In this formulation:
* $Y(1)$ represents the potential outcome measure (e.g., task completion time, error rate) of a post-stroke user performing task $x$.
* $Y(0)$ represents the potential outcome measure of a neurotypical user performing the same task $x$, effectively representing the nominal task difficulty.
* $E[\cdot \mid X=x]$ denotes the expected value conditioned on the task parameters $X$ being equal to $x$.

Our approach estimates the expected outcome for the neurotypical population ($E[Y(0) \mid X=x]$) using data collected from neurotypical users performing task $x$. Similarly, the expected outcome for a post-stroke user ($E[Y(1) \mid X=x]$) is estimated from data specific to that individual post-stroke user performing task $x$.

## Framework Implementation and Data Requirements

Implementing this framework to learn personalized functional difficulty models requires two distinct datasets:

1.  **Limited Mobility / Post-Stroke Dataset:** Contains data from users with limited mobility (e.g., post-stroke individuals) performing the tasks. Example file path: `simplified_data/poststroke_data.csv`.
2.  **Neurotypical Dataset:** Contains data from neurotypical users performing the same tasks. This dataset is used to model the nominal task difficulty. Example file path: `simplified_data/neurotypical_data.csv`.

Both datasets must adhere to a specific structure, including the following essential columns:

* `PID`: A unique identifier for each participant. This is crucial for tracking individual performance and results.
* `Task Parameters`: One or more columns (e.g., `param1`, `param2`, ..., `paramN`) that quantitatively describe the specific configuration or parameters of the task being performed ($X=x$).
* `Outcome Measure`: A single column representing the key performance metric for the task execution (e.g., time taken, success/failure flag, error count) ($Y$).



## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
```
git clone https://github.com/interaction-lab/stroke-exercise-difficulty-estimation.git
cd stroke-exercise-difficulty-estimation
```

2. Install the required dependencies. It is recommended to use a virtual environment:

Using venv:
```
python -m venv .venv
source .venv/bin/activate # On Windows, use `.venv\Scripts\activate`
```

Using conda:
```
conda create -n exercise_difficulty python=3.9
conda activate exercise_difficulty
```

3. Install dependencies
```
pip install -r requirements.txt
```

## Reproducing the Analysis

There are two main ways to reproduce the analysis presented in this paper:

1.  **Evaluate the pre-computed results:** If you want to quickly verify the main results using the data files provided (presumably elsewhere in the repo or linked), run the following command from the top-level project folder (`stroke-exercise-difficulty-estimation`):

    ```
    python -m src.scripts.analyze_test_results
    ```
    This script will process the existing result files and output the key performance metrics or summary tables.

2.  **Re-run the full experiments:** To generate the experiment results from scratch (e.g., to verify the experimental setup or run on different data splits), use this command from the top-level project folder:

    ```
    python -m src.scripts.run_all_experiments
    ```
    **Note:** This process involves training multiple models and evaluating them across different configurations. It is computationally intensive and may take a significant amount of time depending on your hardware.

## Visualizing Data

To visualize the plots used in the paper, you have a few options:

1.  **Visualize Causal Effect, Baseline, and Ground Truth Spatial Plots:**
    To view the 3D spatial graphs showing the estimated causal effect, baseline comparison, and ground truth data points for a specific participant, run the `src.viz.plot_tree` script:

    ```
    python -m src.viz.plot_tree --pid <participant_id>
    ```
    Replace `<participant_id>` with the numerical ID of the participant you want to plot (e.g., `21`). The `--pid` argument is required.

    *Optional arguments for `plot_tree`:* You can customize the plot generation using `--seed`, `--vmin`, `--vmax`, and `-v, --verbose`. Run `python -m src.viz.plot_tree --help` for details.

2.  **Visualize the Full Decision Tree Structure:**
    To visualize the structure of the underlying decision tree model used in the analysis (this might be one of the baseline models or a specific tree model from the pipeline), run the `src.viz.plot_dt` script:

    ```
    python -m src.viz.plot_dt --pid <participant_id>
    ```

    Replace `<participant_id>` with the desired participant's ID. The `--pid` argument is also required for this script. This will generate a graphical representation of the decision tree.

    Check if `plot_dt` has any optional arguments by running:

    ```
    python -m src.viz.plot_dt --help
    ```


## Citations

If you use this code or the concepts from this project in your research, please cite the relevant publications:

- **Causal Trees to Estimate Functional Difficulty** [Dennler 2025](https://arxiv.org/abs/2403.04109)
  ```
  @ARTICLE{dennler2025modeling,
    author={Dennler, Nathaniel and Shi, Zhonghao and Yoo, Uksang and Nikolaidis, Stefanos and Matari{\'c}, Maja J},
    journal={19th IEEE/RAS-EMBS International Conference on Rehabilitation Robotics (ICORR 2025)},
    title={Modeling Personalized Difficulty of Rehabilitation Exercises Using Causal Trees},
    year={2025},
    keywords={Rehabilitation robotics; Assistive robotics; Motor learning; Clinical evaluations},
    publisher={IEEE/RAS-EMBS},
  }
  ```

- **The BARTR Interaction** [Dennler 2023](https://www.science.org/doi/abs/10.1126/scirobotics.adf7723)
  ```
  @ARTICLE{dennler2023metric,
    author={Dennler, Nathaniel and Cain, Amelia and De Guzman, Erica and Chiu, Claudia and Winstein, Carolee J and Nikolaidis, Stefanos and Matari{\'c}, Maja J},
    journal={Science Robotics},
    title={A metric for characterizing the arm nonuse workspace in poststroke individuals using a robot arm},
    year={2023},
    volume={8},
    number={84},
    pages={eadf7723},
    keywords={Rehabilitation robotics; Assistive robotics; Motor learning; Clinical evaluations},
    publisher={American Association for the Advancement of Science},
    doi={10.1126/scirobotics.adf7723}
  }
  ```

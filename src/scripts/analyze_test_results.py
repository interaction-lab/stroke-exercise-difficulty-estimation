import pandas as pd
import pingouin as pg
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def calculate_individual_mse(verbose=False):
    '''
    Calculate the mean squared error (MSE) between ground truth and predicted
    values for each user. These values are averaged across all seeds in the results table.
    '''
    df = pd.read_csv('simplified_data/results.csv')

    means = df.groupby(['model'])['mse'].mean()
    errors = df.groupby(['model'])['mse'].sem()

    # print the mean and standard error for each model
    for model in means.index:
        print(f'MODEL: {model:30}Mean MSE: {means[model]:.3f}, SEM: {errors[model]:.3f}')

    if verbose:
        # print the statistical test results
        print(pg.pairwise_tests(dv='mse', within='model', subject='seed', data=df).round(3))



def calculate_population_r2(verbose=False, plot_models=['CausalForestDML', 'RandomForestRegressor', 'GradientBoostingRegressor']):
    df = pd.read_csv('simplified_data/results.csv')

    # Iterate over each model in the results table
    for model in df['model'].unique():
        preds_all = []
        gt_all = []
        pids_all = []  # To store PIDs corresponding to preds and gt
        r2_scores = []  # To store R2 scores for averaging across seeds
        
        for seed in df['seed'].unique():
            # Filter rows for the current model and seed
            preds = []
            gt = []
            pids = []  # To store PIDs for the current seed
            
            # Filter by model and seed
            for i, row in df[df['seed'] == seed].iterrows():
                if row['model'] == model:
                    pred_row = np.fromstring(row['pred'].strip('[]'), sep=' ')
                    preds.append(pred_row)

                    gt_row = np.fromstring(row['gt'].strip('[]'), sep=' ')
                    gt.append(gt_row)
                    # Repeat the PID for the length of the preds array
                    pids.append([row['pid']] * len(pred_row))
                    
            
            # Flatten the lists of arrays into 1D arrays
            preds = np.concatenate(preds)
            gt = np.concatenate(gt)
            pids = np.concatenate(pids)  # Flatten the PIDs into 1D array
            
            # Compute the R² score for this seed and store it
            score = r2_score(gt, preds)
            r2_scores.append(score)
            
            preds_all.append(preds)
            gt_all.append(gt)
            pids_all.append(pids)
        
        # Calculate the average R² score across seeds
        avg_r2_score = np.mean(r2_scores)
        sem_r2_score = np.std(r2_scores) / np.sqrt(len(r2_scores))
        
        print(f'MODEL: {model:30}Mean R² score: {avg_r2_score:.3f}, SEM: {sem_r2_score:.3f}')
        
        # Scatter plot if the model is 'CausalForestDML'
        if model in plot_models or plot_models == 'all':        
            # Create a scatter plot with color map based on PID
            plt.scatter(gt, preds, c=pids, cmap='viridis', alpha=0.7)
            plt.colorbar(label="PID")  # Add a color bar to indicate PID
            plt.xlabel("Ground Truth (gt)")

            plt.ylabel("Predictions (preds)")
            plt.title(f"Scatter Plot of Predictions vs Ground Truth for {model}")
            
            plt.show()

if __name__ == "__main__":
    print('\n------')
    print('\nCalculating individual MSE across models...\n')
    print('------\n')

    calculate_individual_mse(verbose=False)

    print('\n------')
    print('\nCalculating population r^2 across models...\n')
    print('------\n')
    calculate_population_r2(verbose=False, plot_models=[])
from ..utils import get_train_test_data, EVAL_PIDS
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

#baselines
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
#causal tree, from https://econml.azurewebsites.net/_autosummary/econml.dml.CausalForestDML.html 
from econml.dml import CausalForestDML

#evaluation
from sklearn.metrics import r2_score, mean_squared_error


results = []

# 20 random seeds
for seed in tqdm([111,112,113,114,115]):
    #models with their associated tuned hyperparameters
    models = [
        {'name': 'CausalForestDML', 'model': CausalForestDML(n_estimators=600,min_samples_leaf=10, discrete_treatment=True, random_state=seed)},
        {'name': 'RandomForestRegressor', 'model': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=seed)},
        {'name': 'GradientBoostingRegressor', 'model': GradientBoostingRegressor(n_estimators=50, learning_rate=.01, max_depth=3, random_state=seed)},
        {'name': 'DecisionTreeRegressor', 'model': DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=2, random_state=seed)},
        {'name': 'SVR', 'model': SVR(C=.1, gamma=.0001, kernel='rbf')},
        {'name': 'NearestNeighbors', 'model': KNeighborsRegressor(n_neighbors=15, weights ='uniform', metric='manhattan')},
        {'name': 'MLPRegressor', 'model': MLPRegressor(hidden_layer_sizes=(50,), learning_rate_init=0.001, max_iter=1000, random_state=seed)},
    ]
    np.random.seed(seed)

    datasets = get_train_test_data(seed)
    test_datasets = [dataset for dataset in datasets if dataset['pid'] not in EVAL_PIDS]
    
    
    for model in models:
        for dataset in test_datasets:

            #train the causal forest model jointly
            if model['name'] == 'CausalForestDML':
                X = np.concatenate([dataset['stroke_X_train'], dataset['neuro_X_train']])
                y = np.concatenate([dataset['stroke_y_train'], dataset['neuro_y_train']])

                T = np.zeros(X.shape[0])
                T[:len(dataset['stroke_X_train'])] = 1

                causal_model = copy.deepcopy(model['model'])
                causal_model.fit(Y=y, T=T, X=X)

                preds = causal_model.effect(dataset['stroke_X_test'])
                
            
            #train the two models seperately
            else:
                # Train the stroke model
                stroke_model = copy.deepcopy(model['model'])
                stroke_model.fit(dataset['stroke_X_train'], dataset['stroke_y_train'])

                # Train the neurotypical model
                neurotypical_model = copy.deepcopy(model['model'])
                neurotypical_model.fit(dataset['neuro_X_train'], dataset['neuro_y_train'])

                # Test the models
                preds = stroke_model.predict(dataset['stroke_X_test']) - neurotypical_model.predict(dataset['stroke_X_test'])


            #log the results
            results.append({
                'pid': dataset['pid'],
                'visit': dataset['visit'],
                'model': model['name'],
                'r2': r2_score(dataset['stroke_y_test'], preds),
                'mse': mean_squared_error(dataset['stroke_y_test'], preds),
                'pred': preds,
                'gt': dataset['stroke_y_test'],
                'seed': seed
            })

results = pd.DataFrame(results)
results.to_csv('../simplified_data/results.csv', index=False)


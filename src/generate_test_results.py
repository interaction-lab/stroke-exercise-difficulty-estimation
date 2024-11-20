from utils import get_train_test_data, EVAL_PIDS
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



datasets = get_train_test_data()
test_datasets = [dataset for dataset in datasets if dataset['pid'] not in EVAL_PIDS]

results = []


#models with their assocaited tuned hyperparameters
models = [
    {'name': 'CausalForestDML', 'model': CausalForestDML(n_estimators=100,min_samples_leaf=5, discrete_treatment=True)},
    {'name': 'RandomForestRegressor', 'model': RandomForestRegressor(n_estimators=100, max_depth=100)},
    {'name': 'GradientBoostingRegressor', 'model': GradientBoostingRegressor()},
    {'name': 'DecisionTreeRegressor', 'model': DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=4)},
    {'name': 'SVR', 'model': SVR(C=10, gamma=0.0001, kernel='rbf')},
    {'name': 'NearestNeighbors', 'model': KNeighborsRegressor(n_neighbors=15, weights ='distance', metric='manhattan')},
    {'name': 'MLPRegressor', 'model': MLPRegressor(hidden_layer_sizes=(50, 50), learning_rate_init=0.001, max_iter=500)},
]

for model in models:
    for dataset in tqdm(test_datasets):

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
        })

results = pd.DataFrame(results)
results.to_csv('../simplified_data/results.csv', index=False)


from utils import get_train_test_data, EVAL_PIDS
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import itertools
 
#baselines
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
#causal tree
from econml.dml import CausalForestDML

#evaluation
from sklearn.metrics import r2_score, mean_squared_error

# Hyperparameter grids
param_grids = {
    'CausalForestDML': {
        'n_estimators': [40, 100, 200, 600, 1000],
        'min_samples_leaf': [1, 5, 10],
    },
    'RandomForestRegressor': {
        'n_estimators': [50, 100, 200, 600, 1000],
        'max_depth': [None, 10, 20, 50, 100],
    },
    'GradientBoostingRegressor': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    },
    'DecisionTreeRegressor': {
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVR': {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['linear', 'rbf'],
    },
    'NearestNeighbors': {
        'n_neighbors' : [5,7,9,11,13,15],
        'weights' : ['uniform','distance'],
        'metric' : ['minkowski','euclidean','manhattan']
    },
    'MLPRegressor': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [200, 500, 1000],
    },
}


datasets = get_train_test_data()

test_datasets = [dataset for dataset in datasets if dataset['pid'] in EVAL_PIDS]

results = []

model_classes = {
    'CausalForestDML': CausalForestDML,
    'RandomForestRegressor': RandomForestRegressor,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'SVR': SVR,
    'NearestNeighbors': KNeighborsRegressor,
    'MLPRegressor': MLPRegressor,
    'XGBoost': XGBRegressor,
}
 
models = []

# Loop through each model in param_grids
for model_name, params in param_grids.items():
  
    for param_values in itertools.product(*params.values()):
    
        param_dict = dict(zip(params.keys(), param_values))    
        models.append({
            'name': f"{model_name}_" + "_".join([f"{k}{v}" for k, v in param_dict.items()]),
            'model': model_classes[model_name](**param_dict)
        })

for model in models:
    for dataset in tqdm(test_datasets):

        #train the causal forest model jointly
        if model['name'].startswith("CausalForestDML"):
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
print(results)
results.to_csv('../simplified_data/results_parameters.csv', index=False)





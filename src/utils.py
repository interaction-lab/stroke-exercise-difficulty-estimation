import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EVAL_PIDS = [27,31,37,32,33,23,36] #random split

def get_train_test_data():
    '''
    Returns a list of datasets to use for each participant for each visit. It outputs all of
    these datasets as a dictionary. For the training data, the label is the time_to_reach. For
    testing, we are interested in the averaged value for the gt_difficulty. 
    '''
    stroke_data = pd.read_csv('../simplified_data/poststroke_data.csv')
    # stroke_data = stroke_data.query('time_to_press < 5 and time_to_press > 0')

    neurotypical_data = pd.read_csv('../simplified_data/neurotypical_data.csv')

    datasets = []

    #seperate each data by individual participant
    for pid in stroke_data['pid'].unique():
        pid_data = stroke_data.query(f'pid == {pid}').copy()

        # #further seperate by visit
        # for visit in stroke_data['visit'].unique():
        # visit_data = pid_data.query(f'visit == "{visit}" & time_to_press > 0').copy()
        visit_data = pid_data.query('time_to_press > 0').copy()
        visit_data.dropna(inplace=True)

        #ignore the visit if it has less than 10 data points
        if len(visit_data) < 10:
            continue

        side = visit_data['side'].iloc[0]

        # X = visit_data[['x','y','z']].values
        X = get_features(visit_data)
        y = visit_data[['time_to_press', 'gt_difficulty']].values

        strokeX_train, strokeX_test, strokey_train, strokey_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #get the neurotypical data for this side
        neurotypical_slice = neurotypical_data.query(f'side == "{side}"').copy()
        # X = neurotypical_slice[['x','y','z']].values
        X = get_features(neurotypical_slice)
        y = neurotypical_slice[['time_to_press']].values

        neuroX_train, neuroy_train = X, y

        datasets.append({
            'stroke_X_train': strokeX_train,
            'stroke_X_test': strokeX_test,
            'stroke_y_train': strokey_train[:,0].flatten(),
            'stroke_y_test': strokey_test[:,1].flatten(),
            'neuro_X_train': neuroX_train,
            'neuro_y_train': neuroy_train.flatten(),
            'pid': pid,
            'visit': 0
        })

    return datasets

def get_features(data):
    '''
    Extract the features from the data
    '''
    data['distance'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    data['x^2'] = data['x']**2
    data['y^2'] = data['y']**2
    data['z^2'] = data['z']**2
    
    # Dummy code the 'statement' column
    statement_dummies = pd.get_dummies(data['statement'], prefix='statement')
    
    # Combine the original data with the dummy-coded columns
    data = pd.concat([data, statement_dummies], axis=1)
    
    # Select the relevant columns for features
    features = data[['x', 'y', 'z', 'distance', 'x^2', 'y^2', 'z^2'] + list(statement_dummies.columns)].values
    
    return features
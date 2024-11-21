import os
import pandas as pd
import numpy as np
from tqdm import tqdm



all_data = []
for i in range(10, 21):
    try:
        for file in os.listdir(f'../data/{i}'):
            data = pd.read_csv(f'../data/{i}/{file}')

            to_append = data[['time', 'side','time_to_press','statement','x','y','z']]
            to_append['pid'] = i
            to_append['visit'] = file.split('.')[0]

            all_data.append(to_append)

    except Exception as e:
        print(e)
all_data = pd.concat(all_data)
all_data.to_csv('../simplified_data/neurotypical_data2.csv', index=False)

#load in all the individual data files
all_data = []
for i in range(21, 37):
    try:
        for file in os.listdir(f'../data/{i}'):
            data = pd.read_csv(f'../data/{i}/{file}')

            to_append = data.query('condition == "constrained"') [['time', 'side','time_to_press','statement','x','y','z']]
            to_append['pid'] = i
            to_append['visit'] = file.split('.')[0]

            all_data.append(to_append)

    except Exception as e:
        print(e)

# concatenate all the data together
all_data = pd.concat(all_data)

def average_time_within_radius(df, target_x, target_y, target_z, radius=.05):

    df = df.copy()
    # Compute Euclidean distance
    df['distance'] = np.sqrt(
        (df['x'] - target_x) ** 2 +
        (df['y'] - target_y) ** 2 +
        (df['z'] - target_z) ** 2
    )
    
    # Filter points within 5 cm
    within_radius = df[df['distance'] <= radius]
    within_radius = within_radius[within_radius['time_to_press'] > 0]
    
    # Calculate average time
    average_time = within_radius['time_to_press'].mean()
    
    return average_time

neurotypical_data = pd.read_csv('../simplified_data/neurotypical_data.csv')

gt_times = []
#calculate the gt value for each trial
for i,row in tqdm(all_data.iterrows()):
    # get all points in all data within a 5cm ball from this point
    pid_df = all_data.query(f'pid == {row["pid"]}').copy()
    stroke_time = average_time_within_radius(pid_df, row['x'], row['y'], row['z'])
    neurotypical_time = average_time_within_radius(neurotypical_data, row['x'], row['y'], row['z'])
    gt_times.append(stroke_time - neurotypical_time)

all_data['gt_difficulty'] = gt_times
all_data = all_data[all_data['side'].isin(['l', 'r'])]
all_data.to_csv('../simplified_data/poststroke_data.csv', index=False)
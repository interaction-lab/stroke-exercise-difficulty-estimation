from src.utils import get_features
import numpy as np
import pandas as pd

def get_X():
    '''
    This function generates a bunch of random points that we can color based on the causal effect
    '''
    NUM_SAMPLES = 4_000
    #sampled points
    x_scale, y_scale, z_scale = 60, 30, 40 # in centimeters
    x_offset, y_offset, z_offset = -30, 0, 0 # in centimeters
    max_distance, min_distance = 30, 10 # in centimeters

    # first generate random data within the correct bounds 
    data = np.random.random((NUM_SAMPLES,4)) * [x_scale, y_scale, z_scale, 0] + [x_offset, y_offset, z_offset, 0]
    #select those data that satisfy the space we are looking at
    X = data[(data[:,0]**2 + data[:,1]**2 < max_distance**2) & \
            (data[:,0]**2 + data[:,1]**2 > min_distance**2)]
    X = X/100

    X[:,3] = np.random.choice([0,1,2,3], size=X.shape[0])

    df = pd.DataFrame(X, columns=['x','y','z','statement'])

    return get_features(df, return_names=True)

def get_groups(tree, covariates):
    '''
    This function takes the tree structure we learned, and converts it to a dictionary where
    the key is the leaf id (an integer), and the value is a list of boolean expressions that
    need to be satistfied to be a member of the leaf.
    
    tree - the sklearn tree structure
    covariates - a list of strings in order that correspond to the column names of a df
    '''
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    criteria_dict = {}
    
    stack = [(0, 0, [])]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth, criteria = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1, criteria + [f'{covariates[feature[node_id]]} <= {threshold[node_id]}']))
            stack.append((children_right[node_id], depth + 1, criteria + [f'{covariates[feature[node_id]]} > {threshold[node_id]}']))
        else:
            is_leaves[node_id] = True
            criteria_dict[node_id] = criteria
            
    
    return criteria_dict #return the lists that are not empty from criteria_array

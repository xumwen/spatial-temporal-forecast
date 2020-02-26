import os
import zipfile
import numpy as np
import torch
import pandas as pd
import datetime
import calendar
from dateutil.relativedelta import relativedelta
from math import sin, asin, cos, radians, sqrt

def load_nyc_sharing_bike_data(directory="data/NYC-Sharing-Bike"):
    if (not os.path.isfile(directory + "/adj_mat.npy")
            or not os.path.isfile(directory + "/node_values.npy")):
        if os.path.isfile(directory + "/NYC-Sharing-Bike.zip"):
            with zipfile.ZipFile(directory + "/NYC-Sharing-Bike.zip", 'r') as zip_ref:
                zip_ref.extractall(directory)
        else:
            parse_nyc_sharing_bike_data(directory, "201307-201612-citibike-tripdata.zip")

    A = np.load(directory + "/adj_mat.npy")
    A = A.astype(np.float32)
    # X's shape is (num_nodes, num_features, num_sequence)
    X = np.load(directory + "/node_values.npy")
    X = X.astype(np.float32)
    print('(num_nodes, num_features, num_time_steps) is ', X.shape)
    
    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds

def parse_nyc_sharing_bike_data(directory, filename):
    zip_path = directory + "/" + filename
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(directory)
    
    # zipfile example: 201307-201402-citibike-tripdata.zip
    start_date = datetime.datetime.strptime(filename.split('-')[0], '%Y%m')
    end_date = datetime.datetime.strptime(filename.split('-')[1], '%Y%m')
    month_delta = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month

    max_timestep = (month_delta + 1) * 31 * 24
    max_nodes = 1000
    nodes_info = {}
    # X's shape is (num_nodes, num_features, num_sequence)
    X = np.zeros((max_nodes, 1, max_timestep))
    
    # read monthly data
    timestep_base = 0
    for delta in range(0, month_delta + 1):
        cur_date = start_date + relativedelta(months = delta)
        read_monthly_tripdata(cur_date, nodes_info, X, timestep_base)
        month_day = calendar.monthrange(cur_date.year, cur_date.month)[1]
        timestep_base += 24 * month_day
    
    num_nodes = len(nodes_info)
    print("nodes num is %d." % num_nodes)
    X = X[:num_nodes][:][:timestep_base]
    A = np.zeros((num_nodes, num_nodes))
    
    # calculate adj matrix
    for (id_i, info_i) in  nodes_info.items():
        for (id_j, info_j) in  nodes_info.items():
            idx_i = info_i['index']
            idx_j = info_j['index']
            if idx_i != idx_j:
                A[idx_i][idx_j] = 1 / calculate_distance(info_i['lon'], info_i['lat']
                                                         , info_j['lon'], info_j['lat'])

    # normalize adj matrix
    A = (A.T / A.sum(axis=1)).T
    
    # save as .npy file
    np.save(directory + "/nodes_info.npy", nodes_info)
    np.save(directory + "/adj_mat.npy", A)
    np.save(directory + "/node_values.npy", X)
    return

def read_monthly_tripdata(date, nodes_info, X, timestep_base):
    # path example: 2013-08 - Citi Bike trip data.csv or 201409-citibike-tripdata.csv
    path = directory + "/" + date.strftime("%Y-%m") + " - Citi Bike trip data.csv"
    mod = 1
    if not os.path.exists(path):
        path = directory + "/" + date.strftime("%Y%m") + "-citibike-tripdata.csv"
        mod = 2
    if not os.path.exists(path):
        print("[ERROR]File %s not exists." % path)
        return
    
    data = pd.read_csv(path)
    for _, row in data.iterrows():
        # origin data
        start_station_id = row['start station id']
        end_station_id = row['end station id']
        hour_index = row['stoptime'].find(":")
        if mod == 1:
            stoptime = datetime.datetime.strptime(row['stoptime'][:hour_index], '%Y-%m-%d %H')
        elif mod == 2:
            stoptime = datetime.datetime.strptime(row['stoptime'][:hour_index], '%m/%d/%Y %H')
        
        # add station info
        if not nodes_info.get(start_station_id):
            index = len(nodes_info)
            nodes_info[start_station_id] = {}
            nodes_info[start_station_id]['index'] = index
            nodes_info[start_station_id]['lon'] = row['start station longitude']
            nodes_info[start_station_id]['lat'] = row['start station latitude']
        if not nodes_info.get(end_station_id):
            index = len(nodes_info)
            nodes_info[end_station_id] = {}
            nodes_info[end_station_id]['index'] = index
            nodes_info[end_station_id]['lon'] = row['end station longitude']
            nodes_info[end_station_id]['lat'] = row['end station latitude']
        
        end_station_index = nodes_info[end_station_id]['index']
        timestep = timestep_base + (stoptime.day - 1) * 24 + stoptime.hour
        X[end_station_index][0][timestep] += 1
        
    print("Read %s data successfully." % date.strftime("%Y-%m"))
    return

def hav(theta):
    s = sin(theta / 2)
    return s * s
 
def calculate_distance(lon1, lat1, lon2, lat2):
    '''
    Use haversine formula to calculate distance between two points by longtude and latitude.
    The output is in kilometers
    '''
    EARTH_RADIUS=6371           # radius of the earth is 6371km
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
 
    dlon = abs(lon1 - lon2)
    dlat = abs(lat1 - lat2)
    h = hav(dlat) + cos(lat1) * cos(lat2) * hav(dlon)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
 
    return max(distance, 0.1)

def load_metr_la_data(directory="data/METR-LA"):
    if (not os.path.isfile(directory + "/adj_mat.npy")
            or not os.path.isfile(directory + "/node_values.npy")):
        with zipfile.ZipFile(directory + "/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall(directory)

    A = np.load(directory + "/adj_mat.npy")
    # X's shape is (num_nodes, num_features, num_sequence)
    X = np.load(directory + "/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)
    print('(num_nodes, num_features, num_time_steps) is ', X.shape)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))

import os
import zipfile
import numpy as np
import torch
import pandas as pd
import datetime
import calendar
from dateutil.relativedelta import relativedelta
from math import sin, asin, cos, radians, sqrt


"""
Load NYC-Bike dataset
"""
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
    A = change_avg_degree(A, K=100)
    # normalize adj matrix
    A = A / A.sum(axis=0, keepdims=True)
    # X's shape is (num_nodes, num_features, num_timesteps)
    X = np.load(directory + "/node_values.npy")
    X = X.astype(np.float32)
    print('(num_nodes, num_features, num_timesteps) is ', X.shape)
    
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


"""
Load METR-LA dataset
"""
def load_metr_la_data(directory="data/METR-LA"):
    if (not os.path.isfile(directory + "/adj_mat.npy")
            or not os.path.isfile(directory + "/node_values.npy")):
        with zipfile.ZipFile(directory + "/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall(directory)

    A = np.load(directory + "/adj_mat.npy")
    # X's shape is (num_nodes, num_features, num_timesteps)
    X = np.load(directory + "/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)
    print('(num_nodes, num_features, num_timesteps) is ', X.shape)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds


"""
Load PeMS-M dataset
"""
def load_pems_m_data(directory="data/PeMS-M"):
    adj_path = directory + "/W_228.csv"
    A = np.loadtxt(adj_path, delimiter=',')
    A = A.astype(np.float32)
    A = A / A.sum(axis=1)

    node_path = directory + "/V_228.csv"
    X = np.loadtxt(node_path, delimiter=',')
    X = X.transpose(1, 0)
    X = np.expand_dims(X, axis=1)
    X = X.astype(np.float32)
    print('(num_nodes, num_features, num_timesteps) is ', X.shape)
    
    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds

"""
Load PeMS-D7 dataset
"""
def load_pems_d7_data(directory="data/PeMS-D7"):
    if (not os.path.isfile(directory + "/adj_mat.npy")
            or not os.path.isfile(directory + "/node_values.npy")):
        if os.path.isfile(directory + "/PeMSD7.zip"):
            with zipfile.ZipFile(directory + "/PeMSD7.zip", 'r') as zip_ref:
                zip_ref.extractall(directory)
        else:
            parse_pems_d7_data(directory)
    
    A = np.load(directory + "/adj_mat.npy")
    A = A.astype(np.float32)

    # X's shape is (num_nodes, num_features, num_timesteps)
    X = np.load(directory + "/node_values.npy")
    X = X.astype(np.float32)
    # to avoid OOM and only load a part
    percent = 0.5
    X = X[:, :, :int(percent * X.shape[2])]
    print('(num_nodes, num_features, num_timesteps) is ', X.shape)
    
    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds


def parse_pems_d7_data(directory, meta_path='/d07_text_meta_2019_01_10.txt'):
    if not os.path.isfile(directory + "/raw_adj_mat.npy"):
        calculate_pems_adj(directory + meta_path)
    node_dict = str(np.load(directory + "node_dict.npy", allow_pickle=True))
    node_dict = eval(node_dict)
    node_index = get_pems_node_value(directory, node_dict)
    
    # reload and save adj for valid nodes
    A = np.load(directory + "/raw_adj_mat.npy")
    A = A[node_index][:, node_index]
    np.save(directory + "./adj_mat.npy", A)
    
    return

    
def calculate_pems_adj(meta_path):
    # read metadata
    data = pd.read_table(meta_path, sep='\t', usecols=['ID', 'Latitude', 'Longitude'])
    data = data.dropna(axis=0, how='any').reset_index(drop=True)
    num_nodes = len(data)
    
    # calculate adj matrix
    A = np.zeros((num_nodes, num_nodes))
    node_dict = dict()
    for i in range(num_nodes):
        if i % 10 == 0:
            print("percentage:", i/num_nodes)
        node_i = data.loc[i]
        node_dict[int(node_i['ID'])] = i
        for j in range(i+1, num_nodes):
            node_j = data.loc[j]
            dist = calculate_distance(node_i[2], node_i[1], node_j[2], node_j[1])
            A[i][j] = A[j][i] = dist

    # weighted adj
    A = np.exp(A**2 / -10)
    A[A <= 0.05] = 0
    
    np.save("./node_dict.npy", node_dict)
    np.save("./raw_adj_mat.npy", A)
    
    return


def get_pems_node_value(directory, node_dict, data_dir='./txt'):
    start_date = datetime.datetime.strptime('2019-01-23', '%Y-%m-%d')
    end_date = datetime.datetime.strptime('2019-03-22', '%Y-%m-%d')
    day_delta = (end_date - start_date).days + 1
    day_slots = 12 * 24
    
    num_timesteps = day_delta * day_slots
    num_nodes = len(node_dict)
    num_features = 3
    X = np.zeros((num_timesteps, num_nodes, num_features))
    
    # only use weekday data
    weekday_cnt = 0
    for delta in range(day_delta):
        cur_date = start_date + relativedelta(days = delta)
        if cur_date.isoweekday() <= 5:
            time_start = weekday_cnt * day_slots
            time_end = time_start + day_slots
            X[time_start:time_end] = read_pems_daily_data(data_dir, cur_date, node_dict)
            weekday_cnt += 1
    num_timesteps = weekday_cnt * day_slots
    X = X[:num_timesteps]
    
    # get valid nodes by remove nodes with label nan morn than a certain percentage
    percentage = 0.2
    threshold = num_timesteps * percentage
    nan_cnt = np.count_nonzero(np.isnan(X[:,:,0]), axis=0)
    node_index = np.arange(num_nodes)[nan_cnt<threshold]
    X = X[:, node_index]
    num_nodes = len(node_index)
    np.save(directory + "./valid_nodes.npy", node_index)

    # transpose to (time, node * feature)
    X = X.reshape(num_timesteps, num_nodes * num_features)
    df = pd.DataFrame(X)
    
    # linear interpolate
    df = df.interpolate(method='linear', limit_direction='both', axis=0)
    X = np.array(df).reshape(num_timesteps, num_nodes, num_features)
    
    # output shape is (num_nodes, num_features, num_timesteps)
    X = X.transpose(1, 2, 0)
    np.save(directory + "./node_values.npy", X)
    
    return node_index
    
    
def read_pems_daily_data(data_dir, date, node_dict):
    path = data_dir + '/d07_text_station_5min_' + date.strftime("%Y_%m_%d") + '.txt'
    if not os.path.exists(path):
        print("[ERROR]File %s not exists." % path)
        return
    
    # load data
    cols_index = [0, 1, 9, 10, 11]
    cols_name = ['Time', 'Station', 'Flow', 'Occupancy', 'Speed']
    cols_order = ['Time', 'Station', 'Speed', 'Flow', 'Occupancy']
    data = pd.read_table(path, header=None, sep=',', usecols=cols_index)
    data.columns = cols_name
    data = data[cols_order]
    
    num_nodes = len(node_dict)
    num_timesteps = 12 * 24
    num_feature = 3
    daily_X = np.zeros((num_nodes, num_timesteps, num_feature))
    
    # process
    for node_info in data.groupby('Station'):
        if node_dict.get(node_info[0], -1) == -1:
            print("[ERROR]node %d not in dict." % node_info[0])
            continue
        node_index = node_dict[node_info[0]]
        node_feature = node_info[1].iloc[:, -3:]
        daily_X[node_index][:len(node_feature)] = np.array(node_feature)
    
    print("Read %s data successfully." % date.strftime("%Y-%m-%d"))
    
    return daily_X.transpose(1, 0, 2)


"""
Some general function
"""
def change_avg_degree(A, K=100):
    index = len(A) * K
    threshold = sorted(A.flatten(), reverse=True)[index]
    A[A <= threshold] = 0
    return A


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


def generate_dataset(X, num_timesteps_input, num_timesteps_output, dataset):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_timesteps_input, num_features).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_timesteps_output).
    """
    
    # PeMS only use weekday data and a day contains 288 slots(5min per slot)
    if dataset == "pems" or dataset == "pems-m":
        day_slots = 288
    else:
        day_slots = X.shape[2]

    # Save samples
    features, target = [], []
    for day in range(X.shape[2] // day_slots):
        day_start = day_slots * day
        day_end = day_slots * (day+1)
        X_day = X[:, :, day_start:day_end]
        # Generate the beginning index and the ending index of a sample, which
        # contains (num_points_for_training + num_points_for_predicting) points
        indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
                in range(X_day.shape[2] - (
                    num_timesteps_input + num_timesteps_output) + 1)]
        
        for i, j in indices:
            features.append(
                X_day[:, :, i: i + num_timesteps_input].transpose(
                    (0, 2, 1)))
            target.append(X_day[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))



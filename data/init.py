import csv
from random import shuffle

import numpy as np
import pandas as pd
import requests
import os
from io import StringIO
#from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from envirocar import TrackAPI, DownloadClient, BboxSelector, ECConfig

from constants import all_columns_present_id_list, delimiter_value, download_output_dir, feature_list, \
    original_dataset_csv, original_dataset_features, original_dataset_target, original_delimiter, output_dir, \
    output_dir_2, required_columns, \
    target_value

# create an initial but optional config and an api client
config = ECConfig()
track_api = TrackAPI(api_client=DownloadClient(config=config))

global_track_df_object = None
sequence_len = 10
ids_list = []
def set_trackdf():
    track_df = pd.read_csv('track_df.csv')
    global global_track_df_object
    global_track_df_object = track_df
def get_track_ids():
    track_df = global_track_df_object
    unique_ids = track_df['track.id'].unique()
    return unique_ids

def download_data2():
    track_df = pd.read_csv('track_df2.csv')
    unique_ids = track_df['track.id'].unique()
    endpoint = "https://envirocar.org/api/stable/tracks/"
    bbox = " 7.601165771484375,51.94807412325402,7.648200988769531,51.97261482608728"  # It defines a bbox for the area of Münster (Germany)
    os.makedirs(output_dir_2, exist_ok=True)
    global ids_list
    for track_id in unique_ids:
        url = f'{endpoint}{track_id}.csv'
        response = requests.get(url, headers={})
        if response.status_code == 200:
            # Save the CSV file to the output directory
            content = StringIO(response.content.decode('utf-8'))
            reader = csv.reader(content, delimiter=original_delimiter)
            cleaned_rows = [[item.strip() for item in row] for row in reader]
            #print("cleaned_rows",cleaned_rows[0])
            output_file = os.path.join(download_output_dir, f'{track_id}.csv')
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=original_delimiter)
                writer.writerows(cleaned_rows)
            df = pd.read_csv(output_file, delimiter=original_delimiter)
            if check_columns(df, required_columns):
                ids_list.append(track_id)
            else:
                print("did not match columns", df.columns)
            print(f"CSV file for track ID '{track_id}' downloaded successfully.")
        else:
            print(f"Error downloading CSV file for track ID '{track_id}': {response.text}")
    print("ids_list", ids_list)
def download_data():
    unique_ids = get_track_ids()
    endpoint = "https://envirocar.org/api/stable/tracks/"
    bbox = " 7.601165771484375,51.94807412325402,7.648200988769531,51.97261482608728"  # It defines a bbox for the area of Münster (Germany)
    os.makedirs(output_dir, exist_ok=True)
    global ids_list
    for track_id in unique_ids:
        url = f'{endpoint}{track_id}.csv'
        response = requests.get(url, headers={})
        if response.status_code == 200:
            # Save the CSV file to the output directory
            content = StringIO(response.content.decode('utf-8'))
            reader = csv.reader(content, delimiter=original_delimiter)
            cleaned_rows = [[item.strip() for item in row] for row in reader]
            #print("cleaned_rows",cleaned_rows[0])
            output_file = os.path.join(download_output_dir, f'{track_id}.csv')
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=original_delimiter)
                writer.writerows(cleaned_rows)
            df = pd.read_csv(output_file, delimiter=original_delimiter)
            if check_columns(df, required_columns):
                ids_list.append(track_id)
            else:
                print("did not match columns", df.columns)
            print(f"CSV file for track ID '{track_id}' downloaded successfully.")
        else:
            print(f"Error downloading CSV file for track ID '{track_id}': {response.text}")
    print("ids_list", ids_list)

def divide_track_ids():
    unique_ids = all_columns_present_id_list
    shuffle(unique_ids)
    total = len(unique_ids)
    print("total unique", total)
    train = int(0.8 * total)
    train_ids = unique_ids[:train]
    test_ids = unique_ids[train:]
    return train_ids, test_ids

def divide_sequences(X, Y):
    n = X.shape[0]
    indices = torch.randperm(n)
    shuffled_X = X[indices]
    shuffled_Y = Y[indices]

    train_split = int(0.75 * n)
    validate_split = int(0.125 * n)
    test_split = n - train_split - validate_split

    train_X = shuffled_X[:train_split]
    train_Y = shuffled_Y[:train_split]

    validate_X = shuffled_X[train_split: train_split + validate_split]
    validate_Y = shuffled_Y[train_split: train_split + validate_split]

    test_X = shuffled_X[train_split + validate_split:]
    test_Y = shuffled_Y[train_split + validate_split:]

    return (train_X, train_Y), (test_X, test_Y), (validate_X, validate_Y)

def create_sequences(data, target=target_value, features=feature_list, sequence_length=64):
    X, y = [], []
    total = len(data)
    #print("columns", data.columns)
    #print("features",features)
    for i in range(sequence_length, total):
        if (total-i >= sequence_length): # dropping end sequences
            X.append(data[features].iloc[i-sequence_length:i].values)
            y.append(data[target].iloc[i])
            #print("y", data[target].iloc[i])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def simple_hash_for_fuel_type(fuelType):
    return hash(fuelType) % 10

def read_original_dataset(sequence_len=20):
    #scaler = MinMaxScaler()
    df = pd.read_csv(original_dataset_csv, delimiter=delimiter_value, low_memory=False)

    df['Device_Time'] = pd.to_datetime(df['Device_Time'])
    df.set_index('Device_Time', inplace=True)
    print("hellow")
    X, y = [], []
    #print("original dataset", df.describe())
    for groupid, group_data in df.groupby(['Car_Id', 'Person_Id', 'Trip']):
        group_data = group_data.sort_values('Device_Time')
        # print("groupDtaa", group_data.columns)
        print("car id:::", groupid)
        #inputs = group_data.drop(columns=['Car_Id', 'Person_Id', 'Trip', 'GPS_Time', 'Device_Time'], axis=1)
        inputs = group_data#[original_dataset_features]
        # inputs[original_dataset_features].fillna(method='ffill', inplace=True)
        # inputs[original_dataset_target].fillna(method='ffill', inplace=True)
        targets = group_data[original_dataset_target]
        inputs = inputs[original_dataset_features]
        #inputs[original_dataset_features] = scaler.fit_transform(df[original_dataset_features])
        #print("inputes", inputs.dtypes)

        for i in range(len(group_data) - sequence_len):
            X.append(inputs[i:i + sequence_len].values)
            y.append(targets.iloc[i + sequence_len])

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1), df
def read_data(ids, normalised = False, original=False):
    if original:
        return read_original_dataset()
    track_df = global_track_df_object
    excluded_columns = ['id', 'time', 'track.begin', 'track.end']
    track_df = track_df.drop(columns=excluded_columns, errors='ignore')
    X = []
    y = []
    total_count = 0
    big_df = []
    for track_id in ids:
        csv_file = f'{output_dir}/{track_id}.csv'
        df = pd.read_csv(csv_file,delimiter=delimiter_value)
        df['time'] = pd.to_datetime(df['time'])
        # tr = track_df[track_df['track.id'] == track_id]
        # df['engineDisplacement'] =tr['sensor.engineDisplacement']
        # df['fuelTypeString'] = tr['sensor.fuelType']
        # df['fuelType'] = df['fuelTypeString'].apply(lambda x: simple_hash_for_fuel_type(x))
        df[target_value].fillna(method='ffill', inplace=True)
        # df['engineDisplacement'].fillna(method='ffill', inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)
        big_df.append(df)
        #print("df.decribe", df.describe())
        #break;
        #print("type", type(df))
        total_count += len(df)
        seq_x, seq_y = create_sequences(df)
        X.append(seq_x)
        y.append(seq_y)
        #
        # print("shape sw", seq.shape)
        # for label, content in df.items():
        #     print(f'label: {label}')
        #     print(f'content: {content}', sep='\n')
        # break;
    print("shapes", len(X), len(y), X[0].shape, y[0].shape)
    single_df = pd.concat(big_df)
    big_X = torch.cat(X)
    big_y = torch.cat(y).unsqueeze(1)

    if (not normalised):
        return big_X, big_y, single_df
    else:
        min_vals_X = big_X.min(dim=0, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        max_vals_X = big_X.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]

        min_vals_y = big_y.min(dim=0, keepdim=True)[0]
        max_vals_y = big_y.max(dim=0, keepdim=True)[0]
        normalized_X = (big_X - min_vals_X) / (max_vals_X - min_vals_X)
        normalized_y = (big_y - min_vals_y) / (max_vals_y - min_vals_y)
        #print("single_df", single_df.describe())
        return normalized_X, normalized_y, single_df

def get_columns(df):
    columns = []
    datatype = []
    for column in df:
        columns.append(column)
        datatype.append(df[column].dtype)
    return columns, datatype

def check_columns(df, required_columns):
    return set(required_columns).issubset(df.columns)

def download_track_df():
    bbox = BboxSelector([ # It defines a bbox for the area of Münster (Germany)
        7.58,  # min_x
        51.94,  # min_y
        7.64,  # max_x
        51.98  # max_y
    ])
    track_df = track_api.get_tracks(bbox=bbox, num_results=1000)
    track_df.to_csv('track_df2.csv', index=False)
    return track_df





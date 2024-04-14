import csv
from random import shuffle

import pandas as pd
import requests
import os
from io import StringIO

import torch
from envirocar import TrackAPI, DownloadClient, BboxSelector, ECConfig

from constants import all_columns_present_id_list, feature_list, output_dir, required_columns, target_value

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
            reader = csv.reader(content, delimiter=';')
            cleaned_rows = [[item.strip() for item in row] for row in reader]
            #print("cleaned_rows",cleaned_rows[0])
            output_file = os.path.join(output_dir, f'{track_id}.csv')
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerows(cleaned_rows)
            df = pd.read_csv(output_file, delimiter=';')
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

def create_sequences(data, target=target_value, features=feature_list, sequence_length=20):
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

def read_data(ids):
    track_df = global_track_df_object
    excluded_columns = ['id', 'time', 'track.begin', 'track.end']
    track_df = track_df.drop(columns=excluded_columns, errors='ignore')
    X = []
    y = []
    total_count = 0
    for track_id in ids:
        csv_file = f'{output_dir}/{track_id}.csv'
        df = pd.read_csv(csv_file,delimiter=';')
        df['time'] = pd.to_datetime(df['time'])
        df.fillna(df.mean(numeric_only=True), inplace=True)
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
    return torch.cat(X),torch.cat(y)

def get_columns(df):
    columns = []
    datatype = []
    for column in df:
        columns.append(column)
        datatype.append(df[column].dtype)
    return columns, datatype

def check_columns(df, required_columns):
    return set(required_columns).issubset(df.columns)







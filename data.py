from random import shuffle

import pandas as pd
import requests
import os

from envirocar import TrackAPI, DownloadClient, BboxSelector, ECConfig

# create an initial but optional config and an api client
config = ECConfig()
track_api = TrackAPI(api_client=DownloadClient(config=config))

def get_track_ids():
    track_df = pd.read_csv('track_df.csv')
    unique_ids = track_df['track.id'].unique()
    return unique_ids
def download_data():
    unique_ids = get_track_ids()
    endpoint = "https://envirocar.org/api/stable/tracks/"
    bbox = " 7.601165771484375,51.94807412325402,7.648200988769531,51.97261482608728"  # It defines a bbox for the area of Münster (Germany)
    output_dir = 'envirocar_tracks_csv'
    os.makedirs(output_dir, exist_ok=True)
    for track_id in unique_ids:
        url = f'{endpoint}{track_id}.csv'
        response = requests.get(url, headers={})
        if response.status_code == 200:
            # Save the CSV file to the output directory
            output_file = os.path.join(output_dir, f'{track_id}.csv')
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"CSV file for track ID '{track_id}' downloaded successfully.")
        else:
            print(f"Error downloading CSV file for track ID '{track_id}': {response.text}")

def divide_track_ids():
    unique_ids = get_track_ids()
    shuffle(unique_ids)
    total = len(unique_ids)
    train = int(0.8 * total)
    train_ids = unique_ids[:train]
    test_ids = unique_ids[train:]
    return train_ids, test_ids

def read_data(ids):
    dfs = []
    for track_id in ids:
        csv_file = f'envirocar_tracks_csv/{track_id}.csv'
        df = pd.read_csv(csv_file,delimiter=';')
        dfs.append(df)
    final_df = pd.concat(dfs, ignore_index=True, sort=False)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    final_df = final_df.drop_duplicates()
    final_df = final_df.reindex(sorted(final_df.columns), axis=1)
    final_df = final_df.fillna(final_df.mean()) # TODO: or fill with median after analysis
    return final_df

def get_columns(df):
    columns = []
    datatype = []
    for column in df:
        columns.append(column)
        datatype.append(df[column].dtype)
    return columns, datatype









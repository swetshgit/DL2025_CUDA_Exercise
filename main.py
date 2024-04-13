
from data import divide_track_ids, download_data, read_data

if __name__ == '__main__':
    print("started")
    # df = get_data()
    # get_time_sequences(df)
    # download_data()
    train_ids, test_ids = divide_track_ids()
    train_pd = read_data(train_ids)
    test_pd = read_data(test_ids)
    print("shape::", train_pd.shape, test_pd.shape)

from data import divide_track_ids, download_data, get_columns, read_data

if __name__ == '__main__':
    print("started")
    # df = get_data()
    # get_time_sequences(df)
    # download_data()
    train_ids, test_ids = divide_track_ids()
    train_pd = read_data(train_ids)
    test_pd = read_data(test_ids)
    columsd_tr, _ = get_columns(train_pd)
    columsd_te, _ = get_columns(test_pd)
    print("shape::", train_pd.shape, test_pd.shape)
    print("train", columsd_tr)
    #print("test", columsd_te)
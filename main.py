import sys

from constants import all_columns_present_id_list, feature_list, target_value
from data.init import divide_sequences, divide_track_ids, download_data, download_track_df, get_columns, read_data, \
    set_trackdf
from models.GRU.model import GRUModel


if __name__ == '__main__':
    print("started")
    set_trackdf()
    it_name = ""
    if(len(sys.argv) > 1):
        it_name = sys.argv[1]
    print("it", it_name)
    X, Y, sdf = read_data(all_columns_present_id_list, True)
    print("singleDF", sdf.describe())
    (train_x, train_y), (test_x, test_y) = divide_sequences(X, Y)
    #download_data()
    # train_ids, test_ids = divide_track_ids()
    #
    # train_x, train_y,_ = read_data(train_ids)
    # test_x, test_y,_ = read_data(test_ids)
    print("shape train::", train_x.shape, train_y.shape)
    print("shape test::", test_x.shape, test_y.shape)
    gru_model = GRUModel(train_x.shape[2], batch_size=128, num_layers=5, iteration_name=it_name,cnn=True,cnn_ol=30)
    gru_model.run_model((train_x,train_y), (test_x, test_y))
    #download_track_df()
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
    train_seq, test_seq, valid_seq = divide_sequences(X, Y)
    feature_len = train_seq[0].shape[2]
    #download_data()
    # train_ids, test_ids = divide_track_ids()
    #
    # train_x, train_y,_ = read_data(train_ids)
    # test_x, test_y,_ = read_data(test_ids)
    print("shape train::", train_seq[0].shape, train_seq[1].shape)
    print("shape test::", test_seq[0].shape, test_seq[1].shape)
    gru_model = GRUModel(feature_len, batch_size=256, num_layers=10, hidden_size=128, iteration_name=it_name,cnn=True,cnn_ol=50, lr=0.000001, reg=1e-5, epochs=30)
    gru_model.run_model(train_seq, test_seq, valid_seq)
    #download_track_df()
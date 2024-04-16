from constants import feature_list, target_value
from data.init import divide_track_ids, download_data, get_columns, read_data, set_trackdf
from models.GRU.model import GRUModel

if __name__ == '__main__':
    print("started")
    set_trackdf()
    #download_data()
    train_ids, test_ids,_ = divide_track_ids()
    train_x, train_y,_ = read_data(train_ids[0:50])
    #test_x, test_y = read_data(test_ids[0:20])
    print("shape train::", train_x.shape, train_y.shape)
    #print("shape test::", test_x.shape, test_y.shape)
    # gru_model = GRUModel(train_x.shape[2])
    # gru_model.run_model((train_x,train_y), (test_x, test_y))
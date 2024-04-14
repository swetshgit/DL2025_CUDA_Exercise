
from data import divide_track_ids, download_data, get_columns, read_data, set_trackdf
from models.GRU.model import GRUModel

if __name__ == '__main__':
    print("started")
    set_trackdf()
    #download_data()
    train_ids, test_ids = divide_track_ids()
    train_x, train_y = read_data(train_ids)
    test_x, test_y = read_data(test_ids)
    print("shape train::", train_x.shape, train_y.shape)
    print("shape test::", test_x.shape, test_y.shape)
    gru_model = GRUModel(train_x.shape[2])
    gru_model.train(train_x,train_y)
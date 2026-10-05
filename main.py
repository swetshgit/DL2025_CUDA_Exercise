import sys

from matplotlib import pyplot as plt

from constants import all_columns_present_id_list, feature_list, target_value
from data.init import divide_sequences, divide_track_ids, download_data, download_track_df, get_columns, read_data, \
    set_trackdf
from models.GRU.model import GRUModel

original = False

if __name__ == '__main__':
    # print("started")
    # set_trackdf()
    # it_name = ""
    # if(len(sys.argv) > 1):
    #     it_name = sys.argv[1]
    # print("it", it_name)
    #
    # X, Y, sdf = read_data(all_columns_present_id_list, True, original)
    # print("singleDF", sdf.describe())
    # train_seq, test_seq, valid_seq = divide_sequences(X, Y)
    # feature_len = train_seq[0].shape[2]
    # #download_data()
    # # train_ids, test_ids = divide_track_ids()
    # #
    # # train_x, train_y,_ = read_data(train_ids)
    # # test_x, test_y,_ = read_data(test_ids)
    # print("shape train::", train_seq[0].shape, train_seq[1].shape)
    # print("shape test::", test_seq[0].shape, test_seq[1].shape)
    # gru_model = GRUModel(feature_len, batch_size=128, num_layers=20, hidden_size=128, iteration_name=it_name,cnn=True,cnn_ol=20, lr=0.00001, reg=1e-5, epochs=10)
    # gru_model.run_model(train_seq, test_seq, valid_seq)

    from PIL import Image

    # Load images
    image1 = Image.open('results/GRU/exp7_loss.png')
    image2 = Image.open('results/GRU/exp7_pred.png')

    # Create a figure to display the images
    fig, ax = plt.subplots(1, 2)  # 1 row, 2 columns

    # Display first image
    ax[0].imshow(image1)
    ax[0].set_title('Best Model: Loss')
    ax[0].axis('off')  # Turn off axis numbering

    # Display second image
    ax[1].imshow(image2)
    ax[1].set_title('Best Model: Predictions')
    ax[1].axis('off')  # Turn off axis numbering
    plt.show()
    #download_track_df()
import datetime

import matplotlib.pyplot as plt

from constants import loss_image_dir


def plot_loss(train_losses, val_losses, model_name ="GRU"):
    now = datetime.datetime.now()
    timestamp_string = now.strftime("%Y-%m-%d %H:%M:%S")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_image_dir +"/" +model_name + "_loss_"+timestamp_string+".png")
    #plt.show()

def plot_acc(train_accuracies, valid_accuracies, model_name="GRU"):
    now = datetime.datetime.now()
    timestamp_string = now.strftime("%Y-%m-%d %H:%M:%S")
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_image_dir + "/" + model_name + "_accuracy_" + timestamp_string + ".png")
    # plt.show()


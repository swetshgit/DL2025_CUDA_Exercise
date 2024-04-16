import matplotlib.pyplot as plt
def plot_loss(train_losses, val_losses, model_name ="GRU"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(model_name + "_loss.png")
    #plt.show()
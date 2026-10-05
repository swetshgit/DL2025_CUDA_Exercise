import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.plots import plot_acc, plot_loss

class CNN_GRU(nn.Module):
    def __init__(self, input_channels, cnn_out_channels, hidden_size, num_layers, device, kernel_size=3, stride=1,
                 padding=1):
        super(CNN_GRU, self).__init__()
        self.device = device
        self.cnn_layers = nn.Sequential(
                nn.Conv1d(input_channels, cnn_out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(cnn_out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1),
                nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(cnn_out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1),
                # nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
                # nn.BatchNorm1d(cnn_out_channels),
                # nn.ReLU(),
                # nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.gru = nn.GRU(cnn_out_channels, hidden_size, num_layers, batch_first=True,dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x.to(self.device)
        x = x.permute(0, 2, 1)
        x = self.cnn_layers(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        output = self.linear(out[:, -1, :])

        return output
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.device=device

    def forward(self, x):
        x.to(self.device)
        out, _ = self.gru(x)
        out = self.linear(out[:, -1, :])
        out = out.to(self.device)
        return out

class GRUModel():
    def __init__(self, input_size, hidden_size=64, num_layers=2, lr=0.0001, batch_size=64, iteration_name="", cnn=False,cnn_ol=64, reg=1e-5, epochs=20):
        # Device priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f'[GRUModel] Using device: {self.device}')
        if(cnn):
            self.model = CNN_GRU(input_size,cnn_ol, hidden_size, num_layers, self.device).to(self.device)
        else:
            self.model = GRU(input_size, hidden_size, num_layers,self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=reg)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracy = []
        self.val_accuracy = []
        self.final_accuracy = 0
        self.name = "GRU/" + iteration_name
        self.batch_size = batch_size
        self.epochs = epochs

    def get_dataloader_obj(self, X,y, shuffle=True):
        X = X.to(self.device)
        y = y.to(self.device)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def train(self, dataloader, epoch):
        self.model.train()
        ind = 0
        train_loss = 0
        for inputs, labels in dataloader:
            self.optimizer.zero_grad()
            #print("self.device", self.device)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            print("device:", inputs.device, labels.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            print(f'Ind {ind + 1}, Loss: {loss.item()}')
            self.optimizer.step()
            ind += 1
            train_loss += loss.item() #* inputs.size(0)
        print(f' Train Epoch {epoch + 1}, Loss: {loss.item()}')
        train_loss /= len(dataloader.dataset)
        self.train_losses.append(train_loss)

    def validate(self, dataloader, epoch):
        #print("len(dataloader.dataset)",len(dataloader.dataset))
        self.model.eval()
        val_loss = 0
        device = self.device
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()# * inputs.size(0)
            print(f' Validate Epoch {epoch + 1}, Loss: {loss.item()}')
        val_loss /= len(dataloader.dataset)
        self.val_losses.append(val_loss)

    def test(self, dataloader, tolerance =0.2):
        self.model.eval()
        total_mse = 0
        total = 0
        total_within_tol = 0
        device = self.device
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                #print("outputs shape", outputs)
                mse = nn.functional.mse_loss(outputs, labels, reduction='sum')
                total_mse += mse.item()

                # Calculate how many predictions are within the tolerance range
                within_tolerance = torch.abs(outputs - labels) <= tolerance * torch.abs(labels)
                total_within_tol += within_tolerance.sum().item()
                total += labels.size(0)

            rmse = np.sqrt(total_mse / total)
            accuracy = total_within_tol / total
            return accuracy, rmse


    def plot_analysis(self):
        print("trainloss", self.train_losses)
        print("val loss", self.val_losses)
        plot_loss(self.train_losses,self.val_losses,self.name)
        #plot_acc(self.train_accuracy, self.val_accuracy, self.name)

    def run_model(self, train, test, valid):
        (train_x,train_y) = train
        (test_x, test_y) = test
        (valid_x, valid_y) = valid

        train_dataloader = self.get_dataloader_obj(train_x, train_y, False)
        valid_dataloader = self.get_dataloader_obj(valid_x, valid_y, False)
        test_dataloader = self.get_dataloader_obj(test_x, test_y, False)

        for epoch in range(self.epochs):
            self.train(train_dataloader, epoch)
            self.validate(valid_dataloader, epoch)
            # self.train_accuracy.append(self.test(train_dataloader)[0])
            # self.val_accuracy.append(self.test(valid_dataloader)[0])
            # print("accuracy::", self.train_accuracy[-1], self.val_accuracy[-1])
        self.final_accuracy = self.test(test_dataloader)
        print("final accuracy:::::", self.final_accuracy)
        self.plot_analysis()
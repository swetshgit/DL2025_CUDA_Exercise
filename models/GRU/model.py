import torch
import torch.nn as nn
import torch.optim as optim

from utils.plots import plot_loss

class CNN_GRU(nn.Module):
    def __init__(self, input_channels, cnn_out_channels, hidden_size, num_layers, device, kernel_size=3, stride=1,
                 padding=1):
        super(CNN_GRU, self).__init__()
        self.device = device
        self.cnn_layers = nn.Sequential(
                nn.Conv1d(input_channels, cnn_out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(cnn_out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(cnn_out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(cnn_out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.gru = nn.GRU(cnn_out_channels, hidden_size, num_layers, batch_first=True)
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
    def __init__(self, input_size, hidden_size=64, num_layers=2, lr=0.0001, batch_size=64, iteration_name="", cnn=False,cnn_ol=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        if(cnn):
            self.model = CNN_GRU(input_size,cnn_ol, hidden_size, num_layers, self.device).to(self.device)
        else:
            self.model = GRU(input_size, hidden_size, num_layers,self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.name = "GRU_" + iteration_name
        self.batch_size = batch_size

    def get_dataloader_obj(self, X,y, shuffle=True):
        X = X.to(self.device)
        y = y.to(self.device)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def train(self, dataloader):
        self.model.train()
        ind = 0
        train_loss = 0
        for inputs, labels in dataloader:
            self.optimizer.zero_grad()
            #print("self.device", self.device)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            #print("device:", inputs.device, labels.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            #print(f'Ind {ind + 1}, Loss: {loss.item()}')
            self.optimizer.step()
            ind += 1
            train_loss += loss.item() #* inputs.size(0)
        #print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        train_loss /= len(dataloader.dataset)
        self.train_losses.append(train_loss)

    def validate(self, dataloader, epochs=20):
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

        val_loss /= len(dataloader.dataset)
        self.val_losses.append(val_loss)

    def plot_analysis(self):
        print("trainloss", self.train_losses)
        print("val loss", self.val_losses)
        plot_loss(self.train_losses,self.val_losses,self.name)

    def run_model(self, train, test, epochs =20):
        (train_x,train_y) = train
        (test_x, test_y) = test

        train_dataloader = self.get_dataloader_obj(train_x, train_y, True)
        test_dataloader = self.get_dataloader_obj(test_x, test_y, True)
        for epoch in range(epochs):
            self.train(train_dataloader)
            self.validate(test_dataloader)
        self.plot_analysis()
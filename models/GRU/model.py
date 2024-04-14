import torch
import torch.nn as nn
import torch.optim as optim


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
    def __init__(self, input_size, hidden_size=64, num_layers=2, lr=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = GRU(input_size, hidden_size, num_layers,self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, X, y, epochs=10, batch_size=64):
        X = X.to(self.device)
        y = y.to(self.device)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


        self.model.train()
        for epoch in range(epochs):
            ind = 0
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                #print("self.device", self.device)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                #print("device:", inputs.device, labels.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                print(f'Ind {ind + 1}, Loss: {loss.item()}')
                self.optimizer.step()
                ind += 1
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

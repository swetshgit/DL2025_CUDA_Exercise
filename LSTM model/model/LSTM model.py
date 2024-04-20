##LSTM model##
import torch
import torch.nn as nn
import torch.optim as optim

class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.device = device

    def forward(self, x): #x should be [batch_size, sequence length, input size]
        x = x.to(self.device)
        out, _ = self.lstm(x)
        
        #Passing the output through the linear layer to transform it to desired output size
        out = self.linear(out[:, -1, :])
        out = out.to(self.device)
        return out #output tensor=model's predictions

class LSTMModel():
    def __init__(self, input_size, hidden_size=64, num_layers=2, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Check for Metal Performance Shaders availability on macOS (for M1/M2 chips)
        self.device = torch.device('mps' if torch.backends.mps.is_available() else self.device)
        self.model = LSTM(input_size, hidden_size, num_layers, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss() #Loss function=MSE Loss
        self.train_losses = []
        self.val_losses = []
        self.name = "LSTM"
        
    #training the LSTM using the training data
    def train(self, X, y, epochs=20, batch_size=64):
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)
            average_loss = total_loss / len(dataloader.dataset)
            self.train_losses.append(average_loss)
            print(f'Epoch {epoch + 1}, Training Loss: {average_loss:.4f}')

    def validate(self, X, y, batch_size=64):
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
        average_loss = total_loss / len(dataloader.dataset)
        self.val_losses.append(average_loss)
        print(f'Validation Loss: {average_loss:.4f}')

    def plot_analysis(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_model(self, train, test):
        (train_x, train_y) = train
        (test_x, test_y) = test
        print("Starting training process...")
        self.train(train_x, train_y)
        print("Starting validation process...")
        self.validate(test_x, test_y)
        self.plot_analysis()

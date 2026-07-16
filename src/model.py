import torch.nn as nn

class ModeloLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(in_features=50 , out_features=1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        ultimo_estado = h_n[0]
        return self.linear(ultimo_estado)
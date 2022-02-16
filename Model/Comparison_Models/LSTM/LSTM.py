import torch
import torch.nn as nn

#Cette classe définit le modèles LSTM
class LSTM(nn.Module):
    def __init__(self,number_of_zone_training,seq_len, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.hidden_batch_size = number_of_zone_training
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1 ,self.hidden_batch_size ,self.hidden_layer_size),
                            torch.zeros(1 ,self.hidden_batch_size ,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(self.seq_len ,self.hidden_batch_size, -1),
                                               self.hidden_cell)
        predictions = self.linear(self.hidden_cell[0].view(self.hidden_batch_size, -1))
        return predictions
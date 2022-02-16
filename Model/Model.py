import torch
import torch.nn as nn
import torch.nn.functional as F

# Cette class défini la sous architecture spatial (CNN Loccal) de l'architecture du ST-Forecase
class CNN(nn.Module):
    def __init__(self, img_size, num_filtre, size_filtre, kernel_maxpooling, stride_maxpooling, output_size_linear):
        super(CNN, self).__init__()
        self.num_filtre = num_filtre
        self.conv1 = nn.Conv2d(1, num_filtre, kernel_size=size_filtre)
        self.conv_out = (img_size - size_filtre) + 1
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_maxpooling, stride=stride_maxpooling)
        self.max_pool_out = (self.conv_out - kernel_maxpooling) // stride_maxpooling + 1
        self.fc1 = nn.Linear(num_filtre * self.max_pool_out * self.max_pool_out, output_size_linear)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3, training=True)
        x = x.view(-1, self.num_filtre * self.max_pool_out * self.max_pool_out)
        x = self.fc1(x)
        return x

#Cette classe définit la sous architecture temporelle (LSTM) et Combine les deux sous architecture spatiale et Temporelle
class Combine(nn.Module):
    def __init__(self, img_size, num_filtre, size_filtre, kernel_maxpooling, stride_maxpooling, output_size_linear,
                 hidden_size, output_size_linear_lstm, batsh_size, seq_len,transform):
        super(Combine, self).__init__()
        self.batsh_size = batsh_size
        self.output_size_linear = output_size_linear
        self.seq_len = seq_len
        self.transform = transform
        self.number_of_zone_training = transform.number_of_zone_training

        self.hidden_size = hidden_size
        self.cnn = CNN(img_size, num_filtre, size_filtre, kernel_maxpooling, stride_maxpooling, output_size_linear)

        self.lstm = nn.LSTM(output_size_linear + 25, self.hidden_size)

        self.linear = nn.Linear(self.hidden_size, output_size_linear_lstm)

        self.hidden_cell = (torch.zeros(1, self.batsh_size, self.hidden_size),
                            torch.zeros(1, self.batsh_size, self.hidden_size))

    def forward(self, x, x_external_data):
        c_out = self.cnn(x)
        c_out = self.transform.correctCNNOutPut(c_out)
        c_out = c_out.view(self.seq_len, self.batsh_size, self.output_size_linear)
        ex = (x_external_data).view(self.seq_len, 1, 25)
        data_external_for_reshape = []
        for i in range(self.number_of_zone_training):
            data_external_for_reshape.append(ex)
        data_external_for_reshape = torch.cat(data_external_for_reshape).view(self.seq_len,
                                                                              self.number_of_zone_training, 25)
        input_lstm = torch.cat([c_out, data_external_for_reshape], 2)
        lstm_out, self.hidden_cell = self.lstm(input_lstm, self.hidden_cell)
        predictions = self.linear(self.hidden_cell[0].view(self.batsh_size, -1))
        return predictions
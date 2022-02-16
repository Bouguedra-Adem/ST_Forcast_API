import pandas as pd
import torch
from Model.Comparison_Models.LSTM.LSTM import  LSTM

#Cette classe définit les différentes opérations sur le modèle LSTM
class LSTM_traitement:
    def __init__(self,hash):
        self.seq_len = 4
        self.hash = hash

    def loss_function(self,y, y_hat):
        return torch.sum((y - y_hat) ** 2)

    def train_mode(self,zones):
        date_rng = pd.to_datetime(pd.date_range(start='2018-10-01 02:00:00', end='2019-10-01 02:00:00', freq='H'))
        date_rng_train = pd.to_datetime(pd.date_range(start='2018-10-01 02:00:00', end='2019-08-01 01:00:00', freq='H'))
        date_rng_test = pd.to_datetime(pd.date_range(start='2019-08-01 02:00:00', end='2019-10-01 02:00:00', freq='H'))

        length_training = len(date_rng_train)
        length_test = len(date_rng_test)
        nombre_seq = len(date_rng) - self.seq_len
        nb_seq_training = nombre_seq - len(date_rng_test)

        df = pd.read_csv('./Assets/RequestDataRaw.csv')

        data = df.loc[(df['requested_date'] >= str(date_rng.min())) & (df['requested_date'] <= str(date_rng.max()))][
            ['p_lat', 'p_lng', 'requested_date']]
        data['geohash'] = data.apply(lambda x: self.hash.geohashFunction(x.p_lat, x.p_lng), axis=1)
        data['requested_date'] = pd.to_datetime(data['requested_date'])
        data['requested_date'] = data['requested_date'].dt.strftime('%Y-%m-%d %H:00:00')
        data.requested_date = pd.to_datetime(data.requested_date)
        data = pd.crosstab(data['requested_date'], data['geohash'])
        data = data.reindex(date_rng,fill_value=0)
        data = data.sort_index()
        data_tranined_zones = data[zones]
        inout_seq_all = self.create_inout_sequences_all_zones(data_tranined_zones, self.seq_len,zones)


        model = LSTM(len(zones),self.seq_len)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
        model.train()
        all_losses = []
        for i in range(25):
            for j in range(nb_seq_training):
                labels = []
                seqs = []
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, model.hidden_batch_size, model.hidden_layer_size),
                                     torch.zeros(1, model.hidden_batch_size, model.hidden_layer_size))
                for zn in range(len(zones)):
                    (seq, label) = inout_seq_all[zn][j]
                    seqs.append(seq)
                    labels.append(torch.tensor(label))

                y_preds = model(torch.tensor(seqs).type(torch.float))
                labels = torch.tensor(labels)
                loss_zones = self.loss_function(labels, y_preds)
                all_losses.append(loss_zones.item())

                loss_zones.backward()
                optimizer.step()
            # =========================================VALIDATION============================================
            model.eval()
            y_preds = torch.tensor([])
            labels = []
            for j in range(nb_seq_training, nombre_seq):
                model.hidden_cell = (torch.zeros(1, model.hidden_batch_size, model.hidden_layer_size),
                                     torch.zeros(1, model.hidden_batch_size, model.hidden_layer_size))
                seqs = []

                for zn in range(len(zones)):
                    (seq, label) = inout_seq_all[zn][j]
                    seqs.append(seq)
                    labels.append(torch.tensor(label))

                y_preds = torch.cat([y_preds, model(torch.tensor(seqs).type(torch.float))])

            labels = torch.tensor(labels)

        torch.save(model,'./Assets/model_LSTM_seq={}_pre={}'.format(self.seq_len, self.hash.precision))

    def predit_period(self,data_tranined_zones,zones,predict_seq):
        inout_seq_all = self.create_inout_sequences_all_zones(data_tranined_zones, self.seq_len,zones)
        model = torch.load('./Assets/model_LSTM_seq={}_pre={}'.format(self.seq_len, self.hash.precision))
        model.eval()
        result = {zn: {'Predicted': list(), 'Observed': list()} for zn in zones}

        for i in range(len(predict_seq)):
            model.hidden_cell = (torch.zeros(1, model.hidden_batch_size, model.hidden_layer_size),
                                 torch.zeros(1, model.hidden_batch_size, model.hidden_layer_size))
            seqs = []

            for zn in range(len(zones)):
                (seq, label) = inout_seq_all[zn][i]
                seqs.append(seq)
                result[zones[zn]]['Observed'].append(label[0])

            y_hats = model(torch.tensor(seqs).type(torch.float))
            y_hats = [int(i.item()) for i in list(y_hats.view(1, len(zones))[0])] #int(i.item())

            for zn in range(len(zones)):
                result[zones[zn]]['Predicted'].append(y_hats[zn])

        return result

    def create_inout_sequences(self,input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw):
            train_seq = input_data[i:i + tw]
            train_label = input_data[i + tw:i + tw + 1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    def create_inout_sequences_all_zones(self,data_tranined_zones, tw,zones):
        inout_seq_all = []
        for zn in zones:
            inout_seq_all.append(self.create_inout_sequences(list(data_tranined_zones[zn]), tw))
        return inout_seq_all

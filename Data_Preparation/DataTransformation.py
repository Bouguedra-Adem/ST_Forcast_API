import pandas as pd
import numpy as np
import torch

# Cette classe définit les différentes transformations sur les données pour les rendre exploitable par le modèel ST-Forecast
class Transform:
    def __init__(self,hash,img_size,seq_len,number_of_zone_training):
        self.hash = hash
        self.img_size = img_size
        self.seq_len = seq_len
        self.number_of_zone_training = number_of_zone_training

    def predict_data(self, Data,External_data, predict_seq):
        zones_seq = []
        predict_zones = self.zones()

        #Data = pd.crosstab(data['requested_date'],data['geohash'])
        #Data.index = pd.DatetimeIndex(Data.index)
        #Data = Data.reindex(predict_seq, fill_value=0)

        for zn in predict_zones:
            if zn not in Data.columns:
                Data[str(zn)] = 0

        for zn in predict_zones:
            zones_seq.append(self.seq_of_demand_zone(Data,zn,self.seq_len))

        external_features_seq = torch.FloatTensor(External_data.loc[predict_seq].values).view(1, 8, len(External_data.columns))[0]
        return (torch.cat(zones_seq),external_features_seq)

    def seq_of_demand_zone(self,data,zone,reshape):
        x = list()
        #Data = pd.crosstab(data['requested_date'],data['geohash'])
        #Data.index = pd.DatetimeIndex(Data.index)
        Data = data
        #Data= Data.reindex(self.date_rng, fill_value=0)
        for zn in self.hash.neighbors(zone,self.img_size).reshape(self.img_size * self.img_size,):
            if zn not in Data.columns:
                Data[str(zn)]= 0
        Data = Data[self.hash.neighbors(zone,self.img_size).reshape(self.img_size*self.img_size,)]
        Data = Data.reindex(self.hash.neighbors(zone,self.img_size).reshape(self.img_size * self.img_size,), axis=1)
        for i in Data.index:
            x.append(torch.from_numpy(np.array(Data.loc[i]).reshape(self.img_size,self.img_size)))
        tensor = torch.stack(x)
        return tensor.reshape(reshape,1,self.img_size,self.img_size)

    def create_inout_sequences(self,input_data):
        inout_seq = list()
        tw = self.seq_len
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append((train_seq ,train_label[0][0][self.img_size//2][self.img_size//2]))
        return inout_seq

    def create_inout_sequences_extarnal_data(self,input_data):
        tw = self.seq_len
        inout_seq_external_data = list()
        L = len(input_data)
        for i in range(L-tw):
            train_seq_external_data = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq_external_data.append(train_seq_external_data.values)
        return torch.FloatTensor(inout_seq_external_data).view(L-tw ,self.seq_len,len(input_data.columns))

    def create_data_final_Pred_Period(self,Data,date_rng):
        sequence_data = list()
        list_training_zones = self.zones()
        for zone in list_training_zones:
            sequence_data.append(self.create_inout_sequences(self.seq_of_demand_zone(Data,zone,len(date_rng))))
        return sequence_data

    def create_data_final(self,data,date_rng):
        sequence_data = list()
        Data = pd.crosstab(data['requested_date'],data['geohash'])
        Data.index = pd.DatetimeIndex(Data.index)
        Data = Data.reindex(date_rng, fill_value=0)
        #list_training_zones = list(Data.sum(axis=0).sort_values(axis=0, ascending= False)[0:self.number_of_zone_training].index)
        list_training_zones = self.zones()
        for zone in list_training_zones:
            sequence_data.append(self.create_inout_sequences(self.seq_of_demand_zone(Data,zone,len(date_rng))))
        return sequence_data

    def correctCNNOutPut(self,c_out):
        i = 0
        Container = [list() for j in range(self.seq_len)]
        for v in c_out:
            Container[i % self.seq_len].append(v)
            i = i + 1

        for j in range(self.seq_len):
            Container[j] = torch.cat(Container[j])
        CorectCNN = torch.cat(Container)
        return CorectCNN

    def zones(self):
        zones = pd.read_csv('./Assets/zones.csv')
        return list(zones['geohash_{}'.format(self.hash.precision)])
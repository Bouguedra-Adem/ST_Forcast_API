import numpy as np
import torch
import torch.nn as nn

#Cette classe permet d'entrainner et de sauvgarder le modèle ST-Forecast
class TrainModel:
    def __init__(self,model,learning_rate,nb_seq_training,nombre_seq,epochs,External_Feautre_sequence,seq,number_of_zone_training):
        self.MSE_criterion = nn.MSELoss()
        self.MAE_criterion = nn.L1Loss()
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.nb_seq_training = nb_seq_training
        self.nombre_seq = nombre_seq
        self.epochs = epochs
        self.External_Feautre_sequence = External_Feautre_sequence
        self.seq = seq
        self.number_of_zone_training = number_of_zone_training

    def RMSE_criterion(self,y,y_hats):
        return torch.sqrt(self.MSE_criterion(y,y_hats))

    def Train(self):
        for epoch in range(self.epochs): #entrainner le modèle au tant de fois qu'il y a d'éphoch
            total_loss = list()
            self.model.train()
            for i in range(0, self.nb_seq_training): #Entrainer le modèle sur les sequences d'entrainement
                input_data = list()
                y = list()

                for j in range(self.number_of_zone_training):
                    input_data.append(self.seq[j][i][0])
                    y.append(self.seq[j][i][1])

                x_train_external_data = self.External_Feautre_sequence[i]
                x_train_zones_seq = torch.cat(input_data)
                y_train_zones_seq = torch.from_numpy(np.array(y)).view(self.number_of_zone_training, -1)
                self.model.hidden_cell[0].detach_()
                self.model.hidden_cell[1].detach_()
                self.optimizer.zero_grad()
                output = self.model(x_train_zones_seq.float(), x_train_external_data.float())
                loss = self.RMSE_criterion(output, y_train_zones_seq.float())
                loss.backward(retain_graph=False)
                self.optimizer.step()
                total_loss.append(loss.item())

            #==============================================================================================
            #======================================VALIDATION==============================================
            #==============================================================================================

            total_loss_val_MSE = list()
            total_loss_val_MAE = list()
            plotpred = list()
            ploty = list()
            self.model.eval()
            for i in range(self.nb_seq_training, self.nombre_seq): #Validation du modèle sur les séquences de validation
                input_data = list()
                y = list()
                for j in range(self.number_of_zone_training):
                    input_data.append(self.seq[j][i][0])
                    y.append(self.seq[j][i][1])

                x_train_external_data = self.External_Feautre_sequence[i]
                x_train_zones_seq = torch.cat(input_data)
                y_train_zones_seq = torch.from_numpy(np.array(y)).view(self.number_of_zone_training, -1)
                pred = self.model(x_train_zones_seq.float(), x_train_external_data.float())
                plotpred.append(pred.detach().numpy().reshape(self.number_of_zone_training))
                ploty.append(np.array(y))

                loss_MSE = self.MSE_criterion(pred,y_train_zones_seq.float())
                total_loss_val_MSE.append(loss_MSE.item())
                loss_MAE = self.MAE_criterion(pred,y_train_zones_seq.float())
                total_loss_val_MAE.append(loss_MAE.item())

            if epoch % 1 == 0:
                print('epoch: ', epoch, 'Avrage sequences loss :', (sum(total_loss) / len(total_loss)), epoch,
                      'val RMSE :', (np.sqrt(sum(total_loss_val_MSE) / len(total_loss_val_MSE))), 'val MSE :',
                      (sum(total_loss_val_MSE) / len(total_loss_val_MSE)), 'val MAE :',
                      (sum(total_loss_val_MAE) / len(total_loss_val_MAE)))





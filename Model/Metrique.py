import torch
import torch.nn as nn

#Cette classe définit les différetes métrique utilisées pour l'évalution et la comparison des différents modèles de prédiction.
class metriques:

    def RMSE(self,y, y_hat):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(y, y_hat))
        return loss

    def MSE(self,y, y_hat):
        criterion = nn.MSELoss()
        loss = criterion(y, y_hat)
        return loss

    def MAE(self,y, y_hat):
        MAE = nn.L1Loss()
        return (MAE(y, y_hat))
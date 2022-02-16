from dataclasses import dataclass
from DataModel.GeohashPred import GeohashPred
import uuid
# Cette classe et une dataclasse qui définit la structure des données à envoyer à la couche présentation
# Elle définit la prédiction de la demande dans plusieur zones à une period de temps (plusieurs intervalle)
# avec l'erreur MSE calculer depuis les prédiction de toutes les zones dans toutes les intervalles du temps.
@dataclass
class Prediction:
    model_name : str
    MSE : float
    pred : list()
    id: str = uuid.uuid1().hex

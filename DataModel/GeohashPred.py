from dataclasses import dataclass
import uuid
# Cette classe et une dataclasse qui définit la structure des données à envoyer à la couche présentation
# Elle définit la prédiction de la demande dans une zone à une period de temps (plusieurs intervalle)
@dataclass
class GeohashPred:
    hashcode : str
    predicted: list()
    observed : list()
    MSE : float
    id: str = uuid.uuid1().hex
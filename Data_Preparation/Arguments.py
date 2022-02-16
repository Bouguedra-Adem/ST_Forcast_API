import pandas as pd
# Cette classe définit les attributs globale qu'on va utiliser pour l'entrainemnt des modèle et les intervalle
# de temps de l'entrainnement et du validation.
class Args:
    def __init__(self):
        self.cuda = True
        self.no_cuda = False
        self.seed = 1
        self.epochs = 40
        self.momentum = 0.5
        self.log_interval = 10
        self.img_size = 9
        self.num_filtre = 64
        self.size_filtre = 5
        self.kernel_maxpooling = 2
        self.stride_maxpooling = 2

        self.output_size_linear = 64
        self.hidden_size=16
        self.output_size_linear_lstm = 1
        self.number_of_zone_training=15
        self.learning_rate = 0.00007
        self.batsh_size = self.number_of_zone_training
        self.seq_len=8
        self.hashcode_len = 6

        #===========================================DATA TRAINING / VALIDATION=============================================
        self.date_rng = pd.to_datetime(pd.date_range(start='2018-10-01 02:00:00', end='2019-10-01 02:00:00', freq='H'))
        self.date_rng_train = pd.to_datetime(pd.date_range(start='2018-10-01 02:00:00', end='2019-08-01 01:00:00', freq='H'))
        self.date_rng_test = pd.to_datetime(pd.date_range(start='2019-08-01 02:00:00', end='2019-10-01 02:00:00', freq='H'))

        #==========================================THE LENGTH OF EACH SEQUENCE ============================================
        self.nombre_seq = len(self.date_rng) - self.seq_len
        self.nb_seq_training = self.nombre_seq - len(self.date_rng_test)
        self.fields = ['p_lat', 'p_lng', 'requested_date']

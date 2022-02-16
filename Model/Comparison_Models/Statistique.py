from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults


from datetime import datetime ,timedelta
import pandas as pd

# Cette classe définit les modèles de comparaison statistique ARIMA et VARMA, et les différentes opérations appliquées à ces deux modèles.
class statistiqueModels:
    def __init__(self,train_zones,collect,hash):
        self.train_zones = train_zones
        self.collect = collect
        self.hash = hash
        self.mini = 0
        self.maxi = 0
        ARIMA.__getnewargs__ = self.__getnewargs__
    # Cette méthode permet d'entrainner et de sauvegarder les modèles VARMA et ARIMA
    def ARMA_VARMA_Train(self,numTimeStamsTrain,pred_time_start,pred_time_end):


        startTimeCollect = pred_time_start - timedelta(hours=numTimeStamsTrain)
        endtimestamp_collect = pred_time_end + timedelta(seconds=3599)
        endTimeTrain = pred_time_end - timedelta(hours=1)

        dephasage = pred_time_start.hour


        date_collect = pd.to_datetime(pd.date_range(start=str(startTimeCollect), end=str(endtimestamp_collect), freq='H'))
        train_date_rng = pd.to_datetime(pd.date_range(start=str(startTimeCollect), end=str(endTimeTrain), freq='H'))
        predict_seq = pd.to_datetime(pd.date_range(start=str(pred_time_start), end=str(pred_time_end), freq='H'))

        #DATA CLEANING
        #df = self.collect.Requests(startTimeCollect, endtimestamp_collect)
        df = pd.read_csv('./Assets/RequestDataRaw.csv')
        data = df.loc[(df['requested_date'] >= str(date_collect.min())) & (df['requested_date'] <= str(date_collect.max()))][['p_lat', 'p_lng', 'requested_date']]
        data['geohash'] = data.apply(lambda x: self.hash.geohashFunction(x.p_lat, x.p_lng), axis=1)
        data['requested_date'] = pd.to_datetime(data['requested_date'])
        data['requested_date'] = data['requested_date'].dt.strftime('%Y-%m-%d %H:00:00')
        data.requested_date = pd.to_datetime(data.requested_date)
        data = pd.crosstab(data['requested_date'], data['geohash'])
        Data = data.reindex(date_collect, fill_value=0)
        for zn in self.train_zones:
            if zn not in Data.columns:
                Data[str(zn)]= 0
        Data = Data.sort_index()

        result_ARMA = {zn: {'Predicted': list(), 'Observed': list()} for zn in self.train_zones}
        result_VARMA = {zn: {'Predicted': list(), 'Observed': list()} for zn in self.train_zones}

        DataTraining = Data.loc[train_date_rng][self.train_zones]
        DataTesting = Data.loc[predict_seq][self.train_zones]

        #ARMA
        for zn in self.train_zones:
            series = DataTraining[zn].values
            #dephasage treatemnt : forecast = self.forcastZone(series,len(DataTesting) + dephasage ,zn)
            forecast = self.forcastZone_Train(series, len(DataTesting), zn)
            labels = DataTesting[zn].values
            '''
            predicted = [self.roundAsbInt(i) for i in list(forecast)]
            result_ARMA[zn]['Predicted'] = predicted[dephasage:]
            '''
            result_ARMA[zn]['Predicted'] = [self.roundAsbInt(i) for i in list(forecast)]
            result_ARMA[zn]['Observed'] = list(labels)
            result_VARMA[zn]['Observed'] = list(labels)
        print('===========================================ARMA END========================================')
        '''#VARMA
        DataTrainingVARMA = DataTraining.astype(float)
        data_fit_model = [list(DataTrainingVARMA.iloc[i].values) for i in range(len(DataTrainingVARMA.index))]

        #VARMA TRAINING
        #SAVE
        model = VARMAX(data_fit_model, order=(3, 3))
        model_fit = model.fit(disp=False)
        model_fit.save('./Assets/VARMA/model_all_zones.pkl')


        #LOAD
        #model_fit = VARMAXResults.load('./Assets/VARMA/model_all_zones.pkl')
        #make prediction

        forecast = model_fit.forecast(steps=len(DataTesting))
        #dephasage treatement : forecast = model_fit.forecast(steps=len(DataTesting) + dephasage)
        print('===========================================VARMA END========================================')
        for y_hat in list(forecast):
            y_hat = list(y_hat)
            #y_hat = y_hat[dephasage:]
            for y in range(len(y_hat)):
                result_VARMA[self.train_zones[y]]['Predicted'].append(y_hat[y])'''

        return (result_ARMA,result_VARMA)

    def __getnewargs__(self):
        return ((self.endog), (self.k_lags, self.k_diff, self.k_ma))

    def normalize(self,v):
        return (v-self.mini)/(self.maxi-self.mini)

    def denormalize(self, v_hat):
        return (v_hat * (self.maxi-self.mini)) + self.mini

    #Entrainne les modèles VARMA et ARMA sur des données normatiliser avec MIN/ MAX normalisatoin
    def ARMA_VARMA_Train_Normalize(self,numTimeStamsTrain,pred_time_start,pred_time_end):


        startTimeCollect = pred_time_start - timedelta(hours=numTimeStamsTrain)
        endtimestamp_collect = pred_time_end + timedelta(seconds=3599)
        endTimeTrain = pred_time_end - timedelta(hours=1)

        dephasage = pred_time_start.hour


        date_collect = pd.to_datetime(pd.date_range(start=str(startTimeCollect), end=str(endtimestamp_collect), freq='H'))
        train_date_rng = pd.to_datetime(pd.date_range(start=str(startTimeCollect), end=str(endTimeTrain), freq='H'))
        predict_seq = pd.to_datetime(pd.date_range(start=str(pred_time_start), end=str(pred_time_end), freq='H'))

        #DATA CLEANING
        #df = self.collect.Requests(startTimeCollect, endtimestamp_collect)
        df = pd.read_csv('./Assets/RequestDataRaw.csv')
        data = df.loc[(df['requested_date'] >= str(date_collect.min())) & (df['requested_date'] <= str(date_collect.max()))][['p_lat', 'p_lng', 'requested_date']]
        data['geohash'] = data.apply(lambda x: self.hash.geohashFunction(x.p_lat, x.p_lng), axis=1)
        data['requested_date'] = pd.to_datetime(data['requested_date'])
        data['requested_date'] = data['requested_date'].dt.strftime('%Y-%m-%d %H:00:00')
        data.requested_date = pd.to_datetime(data.requested_date)
        data = pd.crosstab(data['requested_date'], data['geohash'])
        Data = data.reindex(date_collect, fill_value=0)
        for zn in self.train_zones:
            if zn not in Data.columns:
                Data[str(zn)]= 0
        Data = Data.sort_index()

        result_ARMA = {zn: {'Predicted': list(), 'Observed': list()} for zn in self.train_zones}
        result_VARMA = {zn: {'Predicted': list(), 'Observed': list()} for zn in self.train_zones}

        DataTraining = Data.loc[train_date_rng][self.train_zones]
        DataTesting = Data.loc[predict_seq][self.train_zones]

        maxvalues = list()

        for col in DataTraining.columns:
            maxvalues.append(DataTraining[col].max())

        self.maxi = max(maxvalues)

        DataTraining = DataTraining.applymap(lambda x: self.normalize(x))

        #ARMA
        for zn in self.train_zones:
            series = DataTraining[zn].values
            #dephasage treatemnt : forecast = self.forcastZone(series,len(DataTesting) + dephasage ,zn)
            forecast = self.forcastZone_Train(series, len(DataTesting), zn)
            labels = DataTesting[zn].values
            result_ARMA[zn]['Predicted'] = [self.roundAsbInt(self.denormalize(i)) for i in list(forecast)]
            result_ARMA[zn]['Observed'] = list(labels)
            result_VARMA[zn]['Observed'] = list(labels)
        print('===========================================ARMA END========================================')
        '''#VARMA
        DataTrainingVARMA = DataTraining.astype(float)
        data_fit_model = [list(DataTrainingVARMA.iloc[i].values) for i in range(len(DataTrainingVARMA.index))]

        #VARMA TRAINING
        #SAVE
        model = VARMAX(data_fit_model, order=(3, 3))
        model_fit = model.fit(disp=False)
        model_fit.save('./Assets/VARMA/model_all_zones.pkl')


        #LOAD
        #model_fit = VARMAXResults.load('./Assets/VARMA/model_all_zones.pkl')
        #make prediction

        forecast = model_fit.forecast(steps=len(DataTesting))
        #dephasage treatement : forecast = model_fit.forecast(steps=len(DataTesting) + dephasage)
        print('===========================================VARMA END========================================')
        for y_hat in list(forecast):
            y_hat = list(y_hat)
            #y_hat = y_hat[dephasage:]
            for y in range(len(y_hat)):
                result_VARMA[self.train_zones[y]]['Predicted'].append(y_hat[y])'''

        return (result_ARMA,result_VARMA)

    #rendre la valeur absolu et arrandi d'un nombre réel en entré
    def roundAsbInt(self, number):
        return int(round(abs(number)))

    def forcastZone_Train(self,serie,lengthForcast,zn):
        #TRAIN THE MODEL
        model = ARIMA(serie, order=(3,0,3))
        model_fit = model.fit(disp=0)
        model_fit.save('./Assets/ARIMA/hashcode_{}/model_{}.pkl'.format(self.hash.precision,zn))
        #forecast = model_fit.forecast(steps=lengthForcast)[0]

        #LOADING THE MODEL
        #model_fit = ARIMAResults.load('./Assets/ARIMA/hashcode_{}/model_{}.pkl'.format(self.hash.precision,zn))
        forecast = model_fit.forecast(steps=lengthForcast)[0]
        return forecast

    #Permet d'entainner et de sauvegarder les modèles VARMA et ARMA pour toutes les zone sans normalisation
    def ARMA_VARMA_Predict(self,DataTesting,dephasage): #,pred_time_start,pred_time_end
        result_ARMA = {zn: {'Predicted': list(), 'Observed': list()} for zn in self.train_zones}
        result_VARMA = {zn: {'Predicted': list(), 'Observed': list()} for zn in self.train_zones}
        #ARMA
        for zn in self.train_zones:
            forecast = self.forcastZone_Predict(len(DataTesting) + dephasage, zn)
            #labels = [int(i) for i in DataTesting[zn].values]
            labels = map(int,DataTesting[zn].values)
            predicted = [self.roundAsbInt(i) for i in list(forecast)]
            result_ARMA[zn]['Predicted'] = predicted[dephasage:]
            result_ARMA[zn]['Observed'] = list(labels)
            result_VARMA[zn]['Observed'] = list(labels)
        print('===========================================ARMA END========================================')
        '''#VARMA
        DataTrainingVARMA = DataTraining.astype(float)
        data_fit_model = [list(DataTrainingVARMA.iloc[i].values) for i in range(len(DataTrainingVARMA.index))]

        #VARMA TRAINING
        #SAVE
        model = VARMAX(data_fit_model, order=(3, 3))
        model_fit = model.fit(disp=False)
        model_fit.save('./Assets/VARMA/model_all_zones.pkl')


        #LOAD
        #model_fit = VARMAXResults.load('./Assets/VARMA/model_all_zones.pkl')
        #make prediction

        forecast = model_fit.forecast(steps=len(DataTesting))
        #dephasage treatement : forecast = model_fit.forecast(steps=len(DataTesting) + dephasage)
        print('===========================================VARMA END========================================')
        for y_hat in list(forecast):
            y_hat = list(y_hat)
            #y_hat = y_hat[dephasage:]
            for y in range(len(y_hat)):
                result_VARMA[self.train_zones[y]]['Predicted'].append(y_hat[y])'''

        return (result_ARMA,result_VARMA)

    def forcastZone_Predict(self,lengthForcast,zn):
        #LOADING THE MODEL
        model_fit = ARIMAResults.load('./Assets/ARIMA/hashcode_{}/model_{}.pkl'.format(self.hash.precision,zn))
        forecast = model_fit.forecast(steps=lengthForcast)[0]
        return forecast
from Data_Preparation.Arguments import Args
from Data_Preparation.DataCollecting import Collect
from Data_Preparation.DataCleaning import Cleaning
from Data_Preparation.DataTransformation import Transform
from Data_Preparation.HashFunctinos import Hash
from Model.Comparison_Models.LSTM.traitement import  LSTM_traitement

from Model.Comparison_Models.Statistique import statistiqueModels
from Model.Metrique import metriques

from Model.Model import Combine
import torch
from Model.TrainModel import TrainModel
import pandas as pd
import numpy as np
from datetime import datetime ,timedelta
from flask import Flask
import time


'''args = Args()
collect = Collect(args.fields)
hash = Hash(args.hashcode_len)
clean = Cleaning(hash,args.date_rng)
transform = Transform(hash,args.img_size,args.seq_len,args.number_of_zone_training)'''
#model = Combine(args.img_size,args.num_filtre,args.size_filtre,args.kernel_maxpooling,args.stride_maxpooling,args.output_size_linear,args.hidden_size,args.output_size_linear_lstm,args.batsh_size,args.seq_len,transform)

def ceateAllZMPredsPeriod(pred_time_start,pred_time_end,hash_precision,zones):
    args = Args()

    start_collect_ST_Forecast = pred_time_start - timedelta(hours = args.seq_len)
    start_collect_LSTM = pred_time_start - timedelta(hours = 4)
    start_collect_ARMA = pred_time_start
    end_Collect_all = pred_time_end + timedelta(seconds=3599)

    rng_collect_ST_Forecast = pd.to_datetime(pd.date_range(start=str(start_collect_ST_Forecast), end=str(end_Collect_all), freq='H'))
    rng_collect_LSTM = pd.to_datetime(pd.date_range(start=str(start_collect_LSTM), end=str(end_Collect_all), freq='H'))
    rng_collect_ARMA_VARMA = pd.to_datetime(pd.date_range(start=str(start_collect_ARMA), end=str(end_Collect_all), freq='H'))

    rng_pred = pd.to_datetime(pd.date_range(start=str(pred_time_start), end=str(pred_time_end), freq='H'))

    collect = Collect(args.fields)
    hash = Hash(hash_precision)
    clean = Cleaning(hash, rng_collect_ST_Forecast)
    transform = Transform(hash, args.img_size, args.seq_len, args.number_of_zone_training)

    '''pd_zones = pd.read_csv('./Assets/zones.csv')
    zones = list(pd_zones['geohash_{}'.format(args.hashcode_len)])'''

    #COLLECT DATA
    RequestDataRaw = collect.Requests(start_collect_ST_Forecast,end_Collect_all)
    #RequestDataRaw = pd.read_csv('./Assets/RequestDataRaw.csv')
    WeatherDataRaw = collect.Weather()

    '''data = RequestDataRaw.loc[(RequestDataRaw['requested_date'] >= str(rng_collect_ST_Forecast.min())) & (RequestDataRaw['requested_date'] <= str(rng_collect_ST_Forecast.max()))][
        ['p_lat', 'p_lng', 'requested_date']]
    data['geohash'] = data.apply(lambda x: hash.geohashFunction(x.p_lat, x.p_lng), axis=1)
    data['requested_date'] = pd.to_datetime(data['requested_date'])
    data['requested_date'] = data['requested_date'].dt.strftime('%Y-%m-%d %H:00:00')
    data.requested_date = pd.to_datetime(data.requested_date)
    data = pd.crosstab(data['requested_date'], data['geohash'])
    data.index = pd.DatetimeIndex(data.index)
    Data = data.reindex(rng_collect_ST_Forecast, fill_value=0)
    for zn in zones:
        if zn not in Data.columns:
            Data[str(zn)] = 0
    Data = Data.sort_index()'''
    #CLEAN DATA
    Data = clean.clean_Data_For_All(RequestDataRaw, rng_collect_ST_Forecast, zones)
    WeatherClean = clean.Weather_cleaning(WeatherDataRaw)

    Data_ST_Forecase = Data
    Data_LSTM = Data.loc[rng_collect_LSTM][zones]
    Data_ARMA_VARMA = Data.loc[rng_pred][zones]

    dephasage = pred_time_start.hour

    #ST_Forecast
    #TRANSFORM DATA
    request_data_seq = transform.create_data_final_Pred_Period(Data_ST_Forecase, rng_collect_ST_Forecast)
    External_data_seq = transform.create_inout_sequences_extarnal_data(WeatherClean)

    model_loaded = torch.load(
        './Assets/model_seq={}_pre={}_vos={}'.format(args.seq_len, args.hashcode_len, args.img_size))
    model_loaded.eval()
    period_length = len(rng_pred)
    predict_zones = transform.zones()
    result = {zn: {'Predicted': list(), 'Observed': list()} for zn in predict_zones}
    for i in range(period_length):
        input_data = list()
        y = list()
        for j in range(args.number_of_zone_training):
            input_data.append(request_data_seq[j][i][0])
            y.append(request_data_seq[j][i][1])

        x_train_external_data = External_data_seq[i]
        x_train_zones_seq = torch.cat(input_data)

        y_pred = model_loaded(x_train_zones_seq.float(), x_train_external_data.float())
        y_pred = [int(i.item()) for i in list(y_pred.view(1, 15)[0])]

        for zn in range(len(predict_zones)):
            result[predict_zones[zn]]['Predicted'].append(y_pred[zn])
            result[predict_zones[zn]]['Observed'].append(y[zn].item())

    ST_forecast_result = result

    #ARMA_VARMA
    statistiques = statistiqueModels(zones, collect, hash)
    (ARMA_result, VARMA_result) = statistiques.ARMA_VARMA_Predict(Data_ARMA_VARMA, dephasage)

    #LSTM
    trait = LSTM_traitement(hash)
    result_lstm = trait.predit_period(Data_LSTM,zones,rng_pred)

    return (ST_forecast_result,result_lstm,ARMA_result)

def predictPeriod(pred_time_start,pred_time_end):
    args = Args()

    startTimeCollect = pred_time_start - timedelta(hours = args.seq_len)
    endtimestamp_collect = pred_time_end + timedelta(seconds=3599)
    end_time_predict = pred_time_end - timedelta(seconds=1)
    date_collect = pd.to_datetime(pd.date_range(start=str(startTimeCollect), end=str(endtimestamp_collect), freq='H'))
    predict_seq = pd.to_datetime(pd.date_range(start=str(pred_time_start), end=str(pred_time_end), freq='H'))

    collect = Collect(args.fields)
    hash = Hash(args.hashcode_len)
    clean = Cleaning(hash, date_collect)
    transform = Transform(hash, args.img_size, args.seq_len, args.number_of_zone_training)

    #COLLECT DATA
    RequestDataRaw = collect.Requests(startTimeCollect,endtimestamp_collect)
    WeatherDataRaw = collect.Weather()

    #CLEAN DATA
    RequestClean = clean.clean_data(RequestDataRaw)
    WeatherClean = clean.Weather_cleaning(WeatherDataRaw)

    #TRANSFORM DATA
    request_data_seq = transform.create_data_final(RequestClean, date_collect)
    External_data_seq = transform.create_inout_sequences_extarnal_data(WeatherClean)

    model_loaded = torch.load('./Assets/model_seq={}_pre={}_vos={}'.format(args.seq_len, args.hashcode_len, args.img_size))
    model_loaded.eval()
    period_length = len(predict_seq)
    predict_zones = transform.zones()
    result = {zn: {'Predicted': list(), 'Observed': list()} for zn in predict_zones}
    for i in range(period_length):
        input_data = list()
        y = list()
        for j in range(args.number_of_zone_training):
            input_data.append(request_data_seq[j][i][0])
            y.append(request_data_seq[j][i][1])

        x_train_external_data = External_data_seq[i]
        x_train_zones_seq = torch.cat(input_data)

        y_pred = model_loaded(x_train_zones_seq.float(), x_train_external_data.float())
        y_pred = [ int(i.item()) for i in list(y_pred.view(1,15)[0])]

        for zn in range(len(predict_zones)):
            result[predict_zones[zn]]['Predicted'].append(y_pred[zn])
            result[predict_zones[zn]]['Observed'].append(y[zn].item())

    return result

def predictOneTimeStamp(pred_time_stamp):
    args = Args()

    starttimestamp = pred_time_stamp - timedelta(hours = args.seq_len)
    endtimestamp = pred_time_stamp - timedelta(seconds = 1)
    endtimestamp_collect = pred_time_stamp + timedelta(seconds = 3599)
    date_collect = pd.to_datetime(pd.date_range(start=str(starttimestamp), end=str(endtimestamp_collect), freq='H'))
    predict_seq = pd.to_datetime(pd.date_range(start=str(starttimestamp), end=str(endtimestamp), freq='H'))

    collect = Collect(args.fields)
    hash = Hash(args.hashcode_len)
    clean = Cleaning(hash, date_collect)
    transform = Transform(hash,  args.img_size, args.seq_len, args.number_of_zone_training)

    #COLLECT DATA
    RequestDataRaw = collect.Requests(starttimestamp,endtimestamp_collect)
    WeatherDataRaw = collect.Weather()

    #CLEAN DATA
    RequestClean = clean.clean_data(RequestDataRaw)
    WeatherClean = clean.Weather_cleaning(WeatherDataRaw)

    #PREPARE DATA
    Data = pd.crosstab(RequestClean['requested_date'],RequestClean['geohash'])
    Data.index = pd.DatetimeIndex(Data.index)
    Data = Data.reindex(date_collect, fill_value=0)

    (req_seq,external_seq) = transform.predict_data(Data.loc[predict_seq],WeatherClean, predict_seq)
    labls = Data.loc[pred_time_stamp][transform.zones()].values
    model_loaded = torch.load('./Assets/model_seq={}_pre={}_vos={}'.format(args.seq_len, args.hashcode_len, args.img_size))
    model_loaded.eval()

    predict_values = model_loaded(req_seq.float(),external_seq.float())
    predict_list = [ int(i.item()) for i in list(predict_values.view(1,15)[0])]
    predict_zones = transform.zones()
    result = {predict_zones[zn] : {'Predicted': predict_list[zn],'Observed':labls[zn] }for zn in range(len(predict_zones))}
    #print(predict_list)
    #print(list(labls))

def calulateMetrics(result_dict,zones):
    metrics = metriques()

    result = {zn: {'MSE': 0, 'RMSE': 0,'MAE':0} for zn in zones}
    for zn in zones:
        y_hats = torch.tensor(result_dict[zn]['Predicted']).type(torch.float)
        labels = torch.tensor(result_dict[zn]['Observed']).type(torch.float)
        result[zn]['MSE'] = metrics.MSE(y_hats,labels).item()
        result[zn]['RMSE'] = metrics.RMSE(y_hats,labels).item()
        result[zn]['MAE'] = metrics.MAE(y_hats,labels).item()
    return result

def calulateMSEEachZone(result_dict,zones):
    metrics = metriques()
    result = {zn: {'MSE': 0} for zn in zones}
    for zn in zones:
        y_hats = torch.tensor(result_dict[zn]['Predicted']).type(torch.float)
        labels = torch.tensor(result_dict[zn]['Observed']).type(torch.float)
        result[zn]['MSE'] = metrics.MSE(y_hats,labels).item()
    return result

def calulateMSEModel(result_dict,zones):
    metrics = metriques()
    MSE_list = list()
    for zn in zones:
        y_hats = torch.tensor(result_dict[zn]['Predicted']).type(torch.float)
        labels = torch.tensor(result_dict[zn]['Observed']).type(torch.float)
        MSE_list.append(metrics.MSE(y_hats,labels).item())
    return sum(MSE_list)/len(MSE_list)

'''pred_time_start = datetime.strptime('2019-08-01 03:00:00', '%Y-%m-%d %H:%M:%S')
pred_time_end = datetime.strptime('2019-08-5 19:00:00', '%Y-%m-%d %H:%M:%S')'''

#train ARMA
'''pd_zones = pd.read_csv('./Assets/zones.csv')
zones = list(pd_zones['geohash_{}'.format(args.hashcode_len)])
pred_time_start = datetime.strptime('2019-08-01 00:00:00', '%Y-%m-%d %H:%M:%S')
pred_time_end = datetime.strptime('2019-10-01 00:00:00', '%Y-%m-%d %H:%M:%S')
statistiques = statistiqueModels(zones,collect, hash)
(ARMA_result,VARMA_result) = statistiques.ARMA_VARMA_Train(7200,pred_time_start,pred_time_end)
print(ARMA_result)

print('========================================================================================')
print('ARMA',calulateMSEModel(ARMA_result,zones))'''

#pred_time_start = datetime.strptime('2019-08-01 02:00:00', '%Y-%m-%d %H:%M:%S')
#pred_time_end = datetime.strptime('2019-10-01 02:00:00', '%Y-%m-%d %H:%M:%S')

#APP Testing
'''pred_time_start = datetime.strptime('2019-08-01 16:00:00', '%Y-%m-%d %H:%M:%S')
pred_time_end = datetime.strptime('2019-08-5 19:00:00', '%Y-%m-%d %H:%M:%S')

start = time.time()
pd_zones = pd.read_csv('./Assets/zones.csv')
zones = list(pd_zones['geohash_{}'.format(args.hashcode_len)])
(ST_forcast, LSTM,ARMA) = ceateAllZMPredsPeriod(pred_time_start,pred_time_end,6,zones)
end = time.time()

print(ST_forcast)
print('predicted',len(ST_forcast['snd1j0']['Predicted']))
print('observed',len(ST_forcast['snd1j0']['Observed']))
print(LSTM)
print('predicted',len(LSTM['snd1j0']['Predicted']))
print('observed',len(LSTM['snd1j0']['Observed']))
print(ARMA)
print('predicted',len(ARMA['snd1j0']['Predicted']))
print('observed',len(ARMA['snd1j0']['Observed']))
print('=======================================================')
print('ST_forecast',calulateMSEModel(ST_forcast,zones))
print('LSTM',calulateMSEModel(LSTM,zones))
print('ARMA',calulateMSEModel(ARMA,zones))

print(f"Runtime of the program is {end - start}")'''
#TEST
'''#pred_time_start = datetime.strptime('2019-08-01 03:00:00', '%Y-%m-%d %H:%M:%S')
#pred_time_end = datetime.strptime('2019-08-5 19:00:00', '%Y-%m-%d %H:%M:%S')

pd_zones = pd.read_csv('./Assets/zones.csv')
zones = list(pd_zones['geohash_{}'.format(args.hashcode_len)])

pred_time_start = datetime.strptime('2019-08-01 02:00:00', '%Y-%m-%d %H:%M:%S')
pred_time_end = datetime.strptime('2019-10-01 02:00:00', '%Y-%m-%d %H:%M:%S')

trait = LSTM_traitement(hash)
#trait.train_mode(zones)
result_lstm = trait.predit_period(pred_time_start,pred_time_end,zones)
print(result_lstm)
#result = calulateMetrics(result_lstm,zones)
#list_mse = [result[zn]['MSE'] for zn in zones]
#print(sum(list_mse)/len(list_mse))'''

'''
statistiques = statistiqueModels(zones,collect, hash)
(ARMA_result,VARMA_result) = statistiques.ARMA_VARMA_Predict(pred_time_start,pred_time_end)

#DSTNET8RESULT = predictPeriod(pred_time_start,pred_time_end)
print('=========================================ARMA====================================')
metrics_ARIMA = calulateMetrics(ARMA_result,zones)
print(metrics_ARIMA)
print('=========================================VARMA===================================')
#metrics_DSTNET = calulateMetrics(DSTNET8RESULT,zones)
#print(metrics_DSTNET)
'''

'''pred_time_start = datetime.strptime('2019-08-01 02:00:00', '%Y-%m-%d %H:%M:%S')
pred_time_end = datetime.strptime('2019-08-5 16:00:00', '%Y-%m-%d %H:%M:%S')
resultat = predictPeriod(pred_time_start,pred_time_end)
print(resultat)'''

'''predict_zones = transform.zones()
print(predict_zones)

dict = {zn : {'Predicted': list(),'Observed':list() }for zn in predict_zones}
print(dict)'''

#APP
'''app = Flask(__name__)
@app.route('/')
def myfunction():
    return 'hello world !!'

if __name__ == '__main__':
    app.run(debug=True)
    '''


args = Args()
args.hashcode_len = 5
args.epochs = 100
collect = Collect(args.fields)
hash = Hash(args.hashcode_len)
clean = Cleaning(hash,args.date_rng)
transform = Transform(hash,args.img_size,args.seq_len,args.number_of_zone_training)
model = Combine(args.img_size,args.num_filtre,args.size_filtre,args.kernel_maxpooling,args.stride_maxpooling,args.output_size_linear,args.hidden_size,args.output_size_linear_lstm,args.batsh_size,args.seq_len,transform)



#COLLECT DATA
#RequestDataRaw = collect.Requests(args.date_rng[0], args.date_rng[len(args.date_rng) - 1])
RequestDataRaw = pd.read_csv('Assets/RequestDataRaw.csv')
WeatherDataRaw = collect.Weather()

print('======================COLLECTING DATA===============================')
print('======================+++++++++++++++===============================')

#CLEAN DATA
RequestClean = clean.clean_data(RequestDataRaw)
WeatherClean = clean.Weather_cleaning(WeatherDataRaw)
print('==========================CLEAN DATA===============================')
print('======================+++++++++++++++===============================')

#Transforma Data
request_data_seq = transform.create_data_final(RequestClean,args.date_rng)
External_data_seq = transform.create_inout_sequences_extarnal_data(WeatherClean)
print('==========================Transforma Data===============================')
print('========================+++++++++++++++=================================')

#TRAINING
train = TrainModel(model,args.learning_rate,args.nb_seq_training,args.nombre_seq,args.epochs,External_data_seq,request_data_seq,args.number_of_zone_training)

print('===============================Training=================================')
print('========================+++++++++++++++=================================')

train.Train()

torch.save(train.model, './Assets/model_seq={}_pre={}_vos={}'.format(args.seq_len,args.hashcode_len,args.img_size))
print('=======================================SAVED========================================')
#model_loaded = torch.load('./Assets/model_seq={}_pre={}_vos={}'.format(args.seq_len,args.hashcode_len,args.img_size))


'''
#date_rng = pd.to_datetime(pd.date_range(start='2018-10-01 02:00:00', end='2019-10-01 02:00:00', freq='H'))
date_rng = pd.to_datetime(pd.date_range(start='2018-10-01 02:00:00', end='2018-10-05 02:00:00', freq='H'))
fields = ['p_lat', 'p_lng', 'requested_date']
startDate = date_rng[0]
endDate = date_rng[len(date_rng) - 1]

data = dataCollect(startDate, endDate,fields)
print(len(data))
'''
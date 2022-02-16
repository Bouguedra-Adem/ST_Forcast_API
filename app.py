from Data_Preparation.Arguments import Args
from Data_Preparation.DataCollecting import Collect
from Data_Preparation.DataCleaning import Cleaning
from Data_Preparation.DataTransformation import Transform
from Data_Preparation.HashFunctinos import Hash
from Model.Comparison_Models.LSTM.traitement import  LSTM_traitement
from DataModel.GeohashPred import GeohashPred
from DataModel.Prediction import Prediction
from Model.Metrique import metriques
from Model.Comparison_Models.Statistique import statistiqueModels
from Model.Model import Combine
from flask_cors import CORS

import torch
import pandas as pd
from datetime import datetime ,timedelta
from flask import Flask
import time


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

def ceateAllZMPredsPeriod(pred_time_start,pred_time_end,hash_precision,zones):
    args = Args()
    args.hashcode_len = hash_precision

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


    #COLLECT DATA
    RequestDataRaw = collect.Requests(start_collect_ST_Forecast,end_Collect_all)
    #RequestDataRaw = pd.read_csv('./Assets/RequestDataRaw.csv')
    WeatherDataRaw = collect.Weather()

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

def calulateMSEModel(result_dict,zones):
    metrics = metriques()
    MSE_list = list()
    for zn in zones:
        y_hats = torch.tensor(result_dict[zn]['Predicted']).type(torch.float)
        labels = torch.tensor(result_dict[zn]['Observed']).type(torch.float)
        MSE_list.append(metrics.MSE(y_hats,labels).item())
    return sum(MSE_list)/len(MSE_list)

#APP
app = Flask(__name__)
CORS(app)

@app.route('/All_models/<pred_time_start_str>/<pred_time_end_str>/<hashcode_len>')
def PredictPeriodAllModels(pred_time_start_str,pred_time_end_str,hashcode_len):
    metrics = metriques()
    start = time.time()
    pred_time_start = datetime.strptime(pred_time_start_str, '%Y-%m-%d %H:%M:%S')
    pred_time_end = datetime.strptime(pred_time_end_str, '%Y-%m-%d %H:%M:%S')
    #hashcode_len = 6
    pd_zones = pd.read_csv('./Assets/zones.csv')
    zones = list(pd_zones['geohash_{}'.format(hashcode_len)])
    (ST_forcast, LSTM, ARMA) = ceateAllZMPredsPeriod(pred_time_start, pred_time_end, int(hashcode_len), zones)
    end = time.time()

    print(f"Runtime of the program is {end - start}")
    print(ST_forcast)
    print(LSTM)
    print(ARMA)

    print('=================================DATA RESTRUCTURING================================')
    start = time.time()
    pred_ST_forcast = [GeohashPred(zn, ST_forcast[zn]['Predicted'], ST_forcast[zn]['Observed'],metrics.MSE(torch.tensor(ST_forcast[zn]['Predicted']).type(torch.float),torch.tensor(ST_forcast[zn]['Observed']).type(torch.float)).item()) for zn in zones]
    pred_LSTM = [GeohashPred(zn, LSTM[zn]['Predicted'], LSTM[zn]['Observed'],metrics.MSE(torch.tensor(LSTM[zn]['Predicted']).type(torch.float),torch.tensor(LSTM[zn]['Observed']).type(torch.float)).item()) for zn in zones]
    pred_ARMA = [GeohashPred(zn, ARMA[zn]['Predicted'], ARMA[zn]['Observed'],metrics.MSE(torch.tensor(ARMA[zn]['Predicted']).type(torch.float),torch.tensor(ARMA[zn]['Observed']).type(torch.float)).item()) for zn in zones]
    result = {'1' : Prediction('ST_forcast', calulateMSEModel(ST_forcast, zones), pred_ST_forcast),
              '2' : Prediction('LSTM', calulateMSEModel(LSTM, zones), pred_LSTM),
              '3' : Prediction('ARMA', calulateMSEModel(ARMA, zones), pred_ARMA)}
    print(result)
    end = time.time()
    print(f"Runtime of the data restructuring is {end - start}")

    return result

@app.route('/Train_models/<pred_time_start_train>/<pred_time_end_train>/<hashcode_len>/<seq_len>/<img_size>/<epochs>')
def TrainModel(pred_time_start_train,pred_time_end_train,hashcode_len,seq_len,img_size,epochs):
    args = Args()
    args.hashcode_len = hashcode_len
    args.epochs = epochs
    args.seq_len = seq_len
    args.img_size = img_size
    args.date_rng = pd.to_datetime(pd.date_range(start=str(pred_time_start_train), end=str(pred_time_end_train), freq='H'))

    args.nombre_seq = len(args.date_rng) - args.seq_len
    args.nb_seq_training = args.nombre_seq - int(round(len(args.date_rng)*0.2))


    collect = Collect(args.fields)
    hash = Hash(args.hashcode_len)
    clean = Cleaning(hash, args.date_rng)
    transform = Transform(hash, args.img_size, args.seq_len, args.number_of_zone_training)
    model = Combine(args.img_size, args.num_filtre, args.size_filtre, args.kernel_maxpooling, args.stride_maxpooling,
                    args.output_size_linear, args.hidden_size, args.output_size_linear_lstm, args.batsh_size,
                    args.seq_len, transform)

    # COLLECT DATA
    RequestDataRaw = collect.Requests(args.date_rng[0], args.date_rng[len(args.date_rng) - 1])
    #RequestDataRaw = pd.read_csv('Assets/RequestDataRaw.csv')
    WeatherDataRaw = collect.Weather()

    print('======================COLLECTING DATA===============================')
    print('======================+++++++++++++++===============================')

    # CLEAN DATA
    RequestClean = clean.clean_data(RequestDataRaw)
    WeatherClean = clean.Weather_cleaning(WeatherDataRaw)
    print('==========================CLEAN DATA===============================')
    print('======================+++++++++++++++===============================')

    # Transforma Data
    request_data_seq = transform.create_data_final(RequestClean, args.date_rng)
    External_data_seq = transform.create_inout_sequences_extarnal_data(WeatherClean)
    print('==========================Transforma Data===============================')
    print('========================+++++++++++++++=================================')

    # TRAINING
    train = TrainModel(model, args.learning_rate, args.nb_seq_training, args.nombre_seq, args.epochs, External_data_seq,
                       request_data_seq, args.number_of_zone_training)

    print('===============================Training=================================')
    print('========================+++++++++++++++=================================')

    train.Train()

    torch.save(train.model,
               './Assets/model_seq={}_pre={}_vos={}'.format(args.seq_len, args.hashcode_len, args.img_size))
    print('=======================================SAVED========================================')

    return 'Model Trained and saved with the name : model_seq={}_pre={}_vos={}'.format(args.seq_len, args.hashcode_len, args.img_size)

if __name__ == '__main__':
    app.run( port=int("5000"), debug=True)
    #app.run(debug=True)
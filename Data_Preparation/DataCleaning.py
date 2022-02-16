import pandas as pd
import numpy as np
#from polygon_geohasher.polygon_geohasher import geohash_to_polygon

# Cette classe assure le nettoyage  des données de la demandes et les données météoralogiques.
class Cleaning():
  def __init__(self,hash,date_rng):
    self.hash = hash
    self.date_rng = date_rng

  def clean_data(self,df) :
    data=df.loc[(df['requested_date'] >= str(self.date_rng.min())) & (df['requested_date']<= str(self.date_rng.max()))][['p_lat','p_lng','requested_date']]
    data['geohash']=data.apply(lambda x: self.hash.geohashFunction(x.p_lat, x.p_lng), axis=1)
    #data['geometry']=data['geohash'].apply(geohash_to_polygon)
    data['requested_date']=pd.to_datetime(data['requested_date'])
    data['requested_date']=data['requested_date'].dt.strftime('%Y-%m-%d %H:00:00')
    return data

  def clean_Data_For_All(self, RequestDataRaw,rng_collect_ST_Forecast,zones):
    data = RequestDataRaw.loc[(RequestDataRaw['requested_date'] >= str(rng_collect_ST_Forecast.min())) & (RequestDataRaw['requested_date'] <= str(rng_collect_ST_Forecast.max()))][
        ['p_lat', 'p_lng', 'requested_date']]
    data['geohash'] = data.apply(lambda x: self.hash.geohashFunction(x.p_lat, x.p_lng), axis=1)
    data['requested_date'] = pd.to_datetime(data['requested_date'])
    data['requested_date'] = data['requested_date'].dt.strftime('%Y-%m-%d %H:00:00')
    data.requested_date = pd.to_datetime(data.requested_date)
    data = pd.crosstab(data['requested_date'], data['geohash'])
    data.index = pd.DatetimeIndex(data.index)
    Data = data.reindex(rng_collect_ST_Forecast, fill_value=0)
    for zn in zones:
        if zn not in Data.columns:
            Data[str(zn)] = 0
    Data = Data.sort_index()

    return Data

  def Weather_cleaning(self,Weather):
    Weather['Time']=pd.to_datetime(Weather['Time'])
    Weather['Time']=Weather['Time'].dt.strftime('%Y-%m-%d %H:00:00')
    Data_Weather=pd.pivot_table(Weather, values=['T','U','Ff'], index=['Time'], aggfunc=np.mean)
    Data_Weather['Time'] =Data_Weather.index
    Data_Weather['Time']=pd.to_datetime(Data_Weather['Time'])
    Data_Weather['hour_of_day'] = Data_Weather['Time'].dt.hour
    Data_Weather=Data_Weather.drop(['Time'],axis=1)
    dfDummies = pd.get_dummies(Data_Weather['hour_of_day'], prefix = 'HOUR')
    Data_Weather=Data_Weather.drop(['hour_of_day'],axis=1)
    df = pd.concat([Data_Weather, dfDummies], axis=1)
    df = df.rename(columns={"T": "temperature_of_the_day", "U": "Humidity",'Ff':'Wind_speed'})
    df = df.drop(["temperature_of_the_day","Humidity"],axis=1)
    df = df.loc[(df.index >= str(self.date_rng.min())) & (df.index <= str(self.date_rng.max()))]
    df.index = pd.DatetimeIndex(df.index)
    df = df.reindex(self.date_rng, fill_value=0)
    df = df.fillna(0)
    return df
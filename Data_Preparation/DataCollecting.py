import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# Cette classe assure le collecte des données depuis BigQuery
class Collect:
    def __init__(self,fields):
        self.fields = fields

    def Requests(self,startDate, endDate):
        credentials = service_account.Credentials.from_service_account_file('./Assets/BigQueryServiceCredentials.json')
        project_id = 'yassir-carpooling'
        client = bigquery.Client(credentials=credentials, project=project_id)
        query_string = '''
             SELECT {}
             FROM `yassir-carpooling.Intern.alger_trips`
             WHERE requested_date >= DATETIME({},{},{},{},{},{}) AND
             requested_date <= DATETIME({},{},{},{},{},{})
           '''.format(','.join(self.fields),startDate.year, startDate.month, startDate.day, startDate.hour, startDate.minute,startDate.second,
                      endDate.year, endDate.month, endDate.day, endDate.hour, endDate.minute, endDate.second)
        query_job = client.query(query_string)
        rows = query_job.result()
        data = pd.DataFrame(data=[list(x.values()) for x in rows], columns=self.fields)
        return data
   "khadija {}".fromat( 2)
    def Weather(self):
        return pd.read_excel('./Assets/Weather_Data.xls')
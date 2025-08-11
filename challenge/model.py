import pandas as pd
import numpy as np

from typing import Tuple, Union, List
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression

import cloudpickle

from google.cloud import storage



SERVICE_ACCOUNT_JSON = "../../keys/latam-challenge-468711-bb18e10af3de.json"  
GCS_BUCKET_NAME = "ml-model-latam"               
GCS_BLOB_PATH = "models/logreg_v1.pkl"         
GCP_LOCATION = "us-east1"                   

class LatamFeatureEngineer:

    def __init__(self):
        self._top_10_features = [
	    "OPERA_Latin American Wings",
	    "MES_7",
	    "MES_10",
	    "OPERA_Grupo LATAM",
	    "MES_12",
	    "TIPOVUELO_I",
	    "MES_4",
	    "MES_11",
	    "OPERA_Sky Airline",
	    "OPERA_Copa Air"
	]
    
    
    def get_period_day(self,date):
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()

        if(date_time > morning_min and date_time < morning_max):
            return 'mañana'
        elif(date_time > afternoon_min and date_time < afternoon_max):
            return 'tarde'
        elif(
            (date_time > evening_min and date_time < evening_max) or
            (date_time > night_min and date_time < night_max)
        ):
            return 'noche'
            
    def is_high_season(self, fecha):
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)

        if ((fecha >= range1_min and fecha <= range1_max) or
            (fecha >= range2_min and fecha <= range2_max) or
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0
    
    def get_min_diff(self, data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff
    
    def transform(self, data: pd.DataFrame, target_column: str=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
        
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
        
        data['min_diff'] = data.apply(self.get_min_diff, axis = 1)
        
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix = 'MES')],
            axis = 1
            )
	
	
        features = features[self._top_10_features]
	
        if target_column is None:
            return features, None
        else:
            target = data[target_column]
            return features, target
	

	
	
	
	
	

	    

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        self.client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_JSON)
        self.bucket = self.client.bucket(GCS_BUCKET_NAME)
        self.blob = self.bucket.blob(GCS_BLOB_PATH)
        

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        
        feature_enginer = LatamFeatureEngineer()
        
        features, target = feature_enginer.transform(data, target_column)
        
        if target is None:
            return features
	    
        else:
	    
            return features, pd.DataFrame(target)
        


    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        

        n_y0 = len(target[target.iloc[:, 0] == 0])
        n_y1 = len(target[target.iloc[:, 0] == 1])
        
        reg_model_2 = LogisticRegression(class_weight={1: n_y0/len(target), 0: n_y1/len(target)})
        reg_model_2.fit(features, target)

        self._model = reg_model_2
        
        serialized_bytes = cloudpickle.dumps(self._model) 
        self.blob.upload_from_string(serialized_bytes, content_type="application/octet-stream")
        
        return None

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        
        if self._model is None:
            downloaded_bytes = self.blob.download_as_bytes()
            self._model = cloudpickle.loads(downloaded_bytes)
        
        y_hat=self._model.predict(features)
        
        return y_hat.tolist()

        

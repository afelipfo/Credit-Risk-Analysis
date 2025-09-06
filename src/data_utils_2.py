import os
import boto3
import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from difflib import get_close_matches

def get_datasets():
  # Specify the credentials
  ACCESS_KEY = config.ACCESS_KEY
  SECRET_KEY = config.SECRET_KEY

  # Create an S3 client
  s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

  # Specify the bucket name
  bucket_name = 'anyoneai-datasets'

  # List objects in the bucket
  response = s3.list_objects_v2(Bucket=bucket_name, Prefix='credit-data-2010/')

  # Verify if the response contains objects
  archivos = []
  if 'Contents' in response:
      print("Files in the bucket:")
      for obj in response['Contents']:
          archivos.append(obj['Key'])
  else:
      print("No files found in the bucket.")

  # Download files
  for arch in archivos:
      bucket_name = 'anyoneai-datasets'
      if '.' in arch:
          print(arch)
          local_file_path = os.path.join(config.DATASET_ROOT_PATH, os.path.basename(arch))
          print(local_file_path)
          try:
              s3.download_file(bucket_name, arch, local_file_path)
              print(f"Downloaded {arch} to {local_file_path}")
          except Exception as e:
              print(f"Error downloading {arch}: {e}")
  df_train = pd.read_csv(f'{config.DATASET_ROOT_PATH}/PAKDD2010_Modeling_Data.txt', delimiter='\t', encoding='iso-8859-1', header=None)
  header = pd.read_excel(f'{config.DATASET_ROOT_PATH}/PAKDD2010_VariablesList.XLS')
  df_train.columns= header['Var_Title'].values
  df_test = pd.read_csv(f'{config.DATASET_ROOT_PATH}/PAKDD2010_Prediction_Data.txt', delimiter='\t', encoding='iso-8859-1', header=None)
  header = pd.read_excel(f'{config.DATASET_ROOT_PATH}/PAKDD2010_VariablesList.XLS')
  header_list = list(header['Var_Title'].values)
  header_list.pop()
  df_test.columns= header_list
  return df_train, df_test

class ApplyOrdinalEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        conditions = [
            (X[self.column_name] >= 0) & (X[self.column_name] < 1000),
            (X[self.column_name] >= 1000) & (X[self.column_name] < 2000),
            (X[self.column_name] >= 2000) & (X[self.column_name] < 3000),
            (X[self.column_name] >= 3000) & (X[self.column_name] < 4000),
            (X[self.column_name] >= 4000) & (X[self.column_name] < 5000),
            (X[self.column_name] >= 5000)
        ]
        choices = [0, 1, 2, 3, 4, 5]
        X['encoded_income'] = np.select(conditions, choices, default=-1)
        return X



class FixOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_check):
        self.columns_to_check = columns_to_check
        self.bounds_ = {}

    def fit(self, X, y=None):
        for column in self.columns_to_check:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.90)  # Usando el cuartil 0.90 como solicitado
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.bounds_[column] = (lower_bound, upper_bound)
        return self

    def transform(self, X):
        for column, bounds in self.bounds_.items():
            lower_bound, upper_bound = bounds
            # Calcular la mediana sin considerar los valores atípicos actuales
            median = X[(X[column] >= lower_bound) & (X[column] <= upper_bound)][column].median()
            # Imputar los valores atípicos con la mediana
            print(X)
            X.loc[X[column] < lower_bound, column] = median
            X.loc[X[column] > upper_bound, column] = median
        return X


    
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.drop(columns=self.columns, inplace=True)
        return X

    
class DeleteConstantColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop_=[]

    def fit(self, X, y=None):
        self.columns_to_drop_ = X.apply(lambda col: col.nunique() == 1)
        self.columns_to_drop_ = self.columns_to_drop_[self.columns_to_drop_].index.tolist()
        return self

    def transform(self, X):
        X_transformed = X.drop(columns=self.columns_to_drop_, errors='ignore')
        return X_transformed
    

class SexNormalization(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['SEX'] = X['SEX'].apply(lambda x: 'NO_INFORMADO' if x not in ['M', 'F'] else x)
        return X
    
 
class FillWithUnknown(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns=columns
        pass     

    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        for col in self.columns:
            X[col] = X[col].replace({pd.NA: 999, np.nan: 999})
        return X
    

class FillWithMode(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.modes ={}
        pass

    def fit(self, X, y=None):
        for col in self.columns:
            self.modes[col]= X[col].mode()[0]
        return self

    def transform(self, X):
        for col in self.columns:
            X[col].fillna(self.modes[col], inplace=True)
        return X        

    
class ProcessBinaryColumns(BaseEstimator, TransformerMixin):
    def __init__(self,columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):    
        for column in self.columns:
            X[column] = X[column].map({'Y': 1, 'N': 0})
            assert len(X[column].unique())<3, f'Multiples values in binary column {X[column].unique()}'

        return X
    
class FrequencyEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.freq_maps = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.freq_maps[column] = X[column].value_counts(normalize=True).to_dict()
        return self
    
    def transform(self, X):
        for column in self.columns:
            X[column] = X[column].map(self.freq_maps[column]).fillna(0)
        return X

class LabelEncodeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.label_encoders = {}

    def fit(self, X, y=None):
        for column_name in self.columns:
            label_encoder = LabelEncoder()
            label_encoder.fit(X[column_name])
            self.label_encoders[column_name] = label_encoder
        return self

    def transform(self, X):
        # Transformar cada columna especificada utilizando LabelEncoder
        for column_name, label_encoder in self.label_encoders.items():
            # Transformar la columna
            #print(f'here con {column_name} {X}')
            X[column_name] = X[column_name].apply(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else len(label_encoder.classes_))
        return X
    
class OneHotEncodeApplication(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.onehot_encoder = OneHotEncoder()

    def fit(self, X, y=None):
        self.onehot_encoder.fit(X[self.columns])
        return self

    def transform(self, X):
        onehot_encoded = self.onehot_encoder.transform(X[self.columns])
        onehot_df = pd.DataFrame(onehot_encoded.toarray(), columns=self.onehot_encoder.get_feature_names_out(self.columns))
        X.reset_index(drop=True, inplace=True)
        onehot_df.reset_index(drop=True, inplace=True)
        working_df = pd.concat([X, onehot_df], axis=1)
        working_df.drop(self.columns, axis=1, inplace=True)
        return working_df
    

class NormalizationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.min_max_scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.min_max_scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.columns] = self.min_max_scaler.transform(X_copy[self.columns])
        return X_copy
    
class CityOfBirthEnrichment(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = ['CITY_OF_BIRTH_TYPE', 'CITY_OF_BIRTH_REGION', 'STATE_OF_BIRTH', 'CITY_OF_BIRTH']
        self.encoders = {} 

    def fit(self, X, y=None):
        df = pd.read_csv(config.CITIES, sep=';', encoding='latin-1')
        df['Município'] = df['Município'].str.upper()
        df.rename(columns={
            'UF': 'STATE_OF_BIRTH',
            'Município': 'CITY_OF_BIRTH',
            'Porte': 'CITY_OF_BIRTH_TYPE',
            'Região': 'CITY_OF_BIRTH_REGION'
        }, inplace=True)
        self.cities_information = df
        self.states = df['STATE_OF_BIRTH'].tolist()
        for column_name in self.columns:
            label_encoder = LabelEncoder()
            label_encoder.fit(df[column_name])
            self.encoders[column_name] = label_encoder
        return self    
    
    def transform(self, X):
        def process_row(row):
            city = row['CITY_OF_BIRTH'].upper()
            state = row['STATE_OF_BIRTH']
            if state in self.states:
                cities_ = self.cities_information[self.cities_information['STATE_OF_BIRTH'] == state]['CITY_OF_BIRTH']
                best_option = get_close_matches(city, cities_, n=1, cutoff=0.1)
                if len(best_option) > 0:
                    details = self.cities_information[(self.cities_information['STATE_OF_BIRTH'] == state) & (self.cities_information['CITY_OF_BIRTH'] == best_option[0])]
                    if not details.empty:
                        row['CITY_OF_BIRTH'] = best_option[0]
                        row['CITY_OF_BIRTH_TYPE'] = details['CITY_OF_BIRTH_TYPE'].values[0]
                        row['CITY_OF_BIRTH_REGION'] = details['CITY_OF_BIRTH_REGION'].values[0]
                        return row
            # Si no se encuentra una coincidencia o no hay detalles, establecer como desconocido
            row['CITY_OF_BIRTH_TYPE'] = 'Desconocido'
            row['CITY_OF_BIRTH_REGION'] = 'Desconocido'
            return row
        X = X.apply(process_row, axis=1)
        
        for col, encoder in self.encoders.items():
             X[col] = X[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else len(encoder.classes_))
        return X
    
class ResidencialCityEnrichment(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = ['RESIDENCIAL_STATE_TYPE', 'RESIDENCIAL_STATE_REGION', 'RESIDENCIAL_STATE', 'RESIDENCIAL_CITY']
        self.encoders = {} 

    def fit(self, X, y=None):
        df = pd.read_csv(config.CITIES, sep=';', encoding='latin-1')
        df['Município'] = df['Município'].str.upper()
        df.rename(columns={
            'UF': 'RESIDENCIAL_STATE',
            'Município': 'RESIDENCIAL_CITY',
            'Porte': 'RESIDENCIAL_STATE_TYPE',
            'Região': 'RESIDENCIAL_STATE_REGION'
        }, inplace=True)
        self.cities_information = df
        self.states = df['RESIDENCIAL_STATE'].tolist()
        for column_name in self.columns:
            label_encoder = LabelEncoder()
            label_encoder.fit(df[column_name])
            self.encoders[column_name] = label_encoder
        return self    
    
    def transform(self, X):
        def process_row(row):
            city = row['RESIDENCIAL_CITY'].upper()
            state = row['RESIDENCIAL_STATE']
            if state in self.states:
                cities_ = self.cities_information[self.cities_information['RESIDENCIAL_STATE'] == state]['RESIDENCIAL_CITY']
                best_option = get_close_matches(city, cities_, n=1, cutoff=0.1)
                if len(best_option) > 0:
                    details = self.cities_information[(self.cities_information['RESIDENCIAL_STATE'] == state) & (self.cities_information['RESIDENCIAL_CITY'] == best_option[0])]
                    if not details.empty:
                        row['RESIDENCIAL_CITY'] = best_option[0]
                        row['RESIDENCIAL_STATE_TYPE'] = details['RESIDENCIAL_STATE_TYPE'].values[0]
                        row['RESIDENCIAL_STATE_REGION'] = details['RESIDENCIAL_STATE_REGION'].values[0]
                        return row
            # Si no se encuentra una coincidencia o no hay detalles, establecer como desconocido
            row['RESIDENCIAL_STATE_TYPE'] = 'Desconocido'
            row['RESIDENCIAL_STATE_REGION'] = 'Desconocido'
            return row
        X = X.apply(process_row, axis=1)
        
        for col, encoder in self.encoders.items():
             X[col] = X[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else len(encoder.classes_))
        return X    
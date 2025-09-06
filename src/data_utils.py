import os
import boto3
import pandas as pd
from src import config
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

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
  df_test = pd.read_csv('dataset/PAKDD2010_Prediction_Data.txt', delimiter='\t', encoding='iso-8859-1', header=None)
  header = pd.read_excel('dataset/PAKDD2010_VariablesList.XLS')
  header_list = list(header['Var_Title'].values)
  header_list.pop()
  df_test.columns= header_list
  return df_train, df_test

def clean_unnecesary_columns(app_train, unnecesary_columns):
  app_train.drop(columns=unnecesary_columns, inplace=True)
  return app_train


def rename_columns_duplicated(app):
  # Get the column names
  columns = app.columns.tolist()

  # Find the index of the second occurrence of the column name you want to change
  index_second_occurrence = columns.index('EDUCATION_LEVEL', columns.index('EDUCATION_LEVEL') + 1)

  # Change the name of the second occurrence of the column
  columns[index_second_occurrence] = 'META_EDUCATION_LEVEL'

  # Update the DataFrame with the new column names
  app.columns = columns
  return  app


def one_hot_encoder_application(app, columns):
    onehot_encoder = OneHotEncoder()
    onehot_encoder.fit(app[columns])
    onehot_encoded = onehot_encoder.transform(app[columns])
    onehot_df = pd.DataFrame(onehot_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(columns))
    app.reset_index(drop=True, inplace=True)
    onehot_df.reset_index(drop=True, inplace=True)
    working_df = pd.concat([app, onehot_df], axis=1)
    working_df = working_df.drop(columns, axis=1)
    return working_df


def normalization(app, ALL_COLUMNS):
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(app[ALL_COLUMNS])
    app[ALL_COLUMNS] = min_max_scaler.transform(app[ALL_COLUMNS])
    app[ALL_COLUMNS] = min_max_scaler.transform(app[ALL_COLUMNS])
    app[ALL_COLUMNS] = min_max_scaler.transform(app[ALL_COLUMNS])
    return app

def fix_outliers(app, columns_to_check):
    """
    Corrige los valores atípicos en las columnas especificadas de un DataFrame
    mediante la imputación con la mediana.

    Parámetros:
    df: DataFrame de pandas.
    columns_to_check: Lista de nombres de columnas para verificar y corregir los valores atípicos.

    Devuelve: DataFrame con valores atípicos corregidos.
    """
    for column in columns_to_check:
        Q1 = app[column].quantile(0.25)
        Q3 = app[column].quantile(0.90)  # Usando el cuartil 0.90 como solicitado
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Calcular la mediana sin considerar los valores atípicos actuales
        median = app[(app[column] >= lower_bound) & (app[column] <= upper_bound)][column].median()
        # Imputar los valores atípicos con la mediana
        app.loc[app[column] < lower_bound, column] = median
        app.loc[app[column] > upper_bound, column] = median
    return app

def preprocess_binary_columns(app, binary_columns):
    """
    Preprocess binary columns by converting "yes" and "no" values to 1 and 0, respectively.

    Parameters:


    df (DataFrame): The DataFrame containing the dataset.
        binary_columns (list): The list of names of binary columns to be preprocessed.


    Returns:


    df_processed (DataFrame): The DataFrame with binary columns preprocessed."""


    # Make a copy of the DataFrame

    # Map "yes" to 1 and "no" to 0 for each binary column
    for column in binary_columns:
        app[column] = app[column].map({'Y': 1, 'N': 0})
        assert len(app[column].unique())<3, f'Multiples values in binary column {app[column].unique()}'

    return app

def frequency_encoding(app, columns):
    """
    Encode categorical columns using frequency encoding.

    Parameters:


    df (DataFrame): The DataFrame containing the dataset.
        columns (list): The list of names of columns to be encoded.


    Returns:


    df (DataFrame): The DataFrame with the encoded columns."""
    # Iterate through each specified column
    for column in columns:
        # Calculate the frequency of each category in the column
        freq = app[column].value_counts(normalize=True)

        # Map each category to its frequency
        app[column + '_freq_encoded'] = app[column].map(freq)

    app.drop(columns=columns,inplace=True)

    return app

def label_encode_columns(app, column_names):
    """
    Encode categorical columns in a DataFrame using LabelEncoder.

    Parameters:


    df (DataFrame): The DataFrame containing the dataset.
        column_names (list): The list of names of columns to be encoded.


    Returns:


    df (DataFrame): The DataFrame with the encoded columns."""

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform each specified column using LabelEncoder
    for column_name in column_names:
        app[column_name + '_encoded'] = label_encoder.fit_transform(app[column_name])

    app.drop(columns=column_names,inplace=True)
    return app

def delete_constant_columns(app):
  columns_to_drop = [col for col in app.columns if app[col].nunique() == 1]
  print(f'duplicated {columns_to_drop}')
  app = app.drop(columns=columns_to_drop, axis=1)
  return app

def sex_normalization(app):
    app['SEX'] = app['SEX'].apply(lambda x: 'NO_INFORMADO' if x not in ['M', 'F'] else x)
    return app


def preprocess_binary_columns(app, binary_columns):
    """
    Preprocess binary columns by converting "yes" and "no" values to 1 and 0, respectively.

    Parameters:


    df (DataFrame): The DataFrame containing the dataset.
        binary_columns (list): The list of names of binary columns to be preprocessed.


    Returns:


    df_processed (DataFrame): The DataFrame with binary columns preprocessed."""


    # Make a copy of the DataFrame

    # Map "yes" to 1 and "no" to 0 for each binary column
    for column in binary_columns:
        app[column] = app[column].map({'Y': 1, 'N': 0})
        assert len(app[column].unique())<3, f'Multiples values in binary column {app[column].unique()}'

    return app


def frequency_encoding(app, columns):
    """
    Encode categorical columns using frequency encoding.

    Parameters:


    df (DataFrame): The DataFrame containing the dataset.
        columns (list): The list of names of columns to be encoded.


    Returns:


    df (DataFrame): The DataFrame with the encoded columns."""
    # Iterate through each specified column
    for column in columns:
        # Calculate the frequency of each category in the column
        freq = app[column].value_counts(normalize=True)

        # Map each category to its frequency
        app[column + '_freq_encoded'] = app[column].map(freq)

    app.drop(columns=columns,inplace=True)

    return app


def label_encode_columns(app, column_names):
    """
    Encode categorical columns in a DataFrame using LabelEncoder.

    Parameters:


    df (DataFrame): The DataFrame containing the dataset.
        column_names (list): The list of names of columns to be encoded.


    Returns:


    df (DataFrame): The DataFrame with the encoded columns."""

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform each specified column using LabelEncoder
    for column_name in column_names:
        app[column_name + '_encoded'] = label_encoder.fit_transform(app[column_name])

    app.drop(columns=column_names,inplace=True)
    return app

def fill_missing_with_unknown(app, columns):

    for col in columns:
        app[col] = app[col].fillna('Unknown').astype(str)

    return app

def complete_with_mode(app, columns):
    for col in columns:
        mode_value = app[col].mode()[0]
        app[col].fillna(mode_value, inplace=True)
    return app
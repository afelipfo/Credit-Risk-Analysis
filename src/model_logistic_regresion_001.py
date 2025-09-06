import data_utils_2
from sklearn.model_selection import train_test_split
from  pipeline_one import PipelineOne
from sklearn.linear_model import LogisticRegression
import joblib
from config import MODEL_ROOT_PATH, SRC_PATH
import shutil

app_train_copy, app_test_copy = data_utils_2.get_datasets()

app_train_x, app_test_x, app_train_y, app_test_y = train_test_split(app_train_copy.drop('TARGET_LABEL_BAD=1', axis=1), app_train_copy['TARGET_LABEL_BAD=1'], test_size=0.2, random_state=42)
print('Got data')
columns = app_train_x.columns.tolist()
index_second_occurrence = columns.index('EDUCATION_LEVEL', columns.index('EDUCATION_LEVEL') )
columns[index_second_occurrence] = 'META_EDUCATION_LEVEL'
app_train_x.columns = columns
app_test_x.columns = columns
app_train_x.drop('META_EDUCATION_LEVEL', inplace=True, axis=1)
app_test_x.drop('META_EDUCATION_LEVEL', inplace=True, axis=1)

pipeline_one = PipelineOne()
pipeline = pipeline_one.pipeline

res = pipeline.fit_transform(app_train_x)
print('Pipeline created and fitted succesfully')
model_ = LogisticRegression()
model_.fit(res, app_train_y)
print('Model created and fitted sucessully')
joblib.dump(pipeline, f'{MODEL_ROOT_PATH}/pipeline_one.pkl')
joblib.dump(model_, f'{MODEL_ROOT_PATH}/001_model_logistic_regresion.pkl')
shutil.copyfile(f'{SRC_PATH}/pipeline_one.py', f'{MODEL_ROOT_PATH}/pipeline_one.py')
print('Model and pipeline save sucessfully')
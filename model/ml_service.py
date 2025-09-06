import json
import time
import joblib
import redis
from sklearn.linear_model import LogisticRegression
import pandas as pd
import config

lr = LogisticRegression()  # Placeholder for your model

# Connect to Redis and assign to variable `db`

db = redis.Redis(host=config.REDIS_IP,
                 port=config.REDIS_PORT,
                 db=config.REDIS_DB_ID,
                 decode_responses=True)

model_logistic_regresion_001 = joblib.load('001_model_logistic_regresion.pkl')
model_logistic_regresion_002 = joblib.load('002_model_logistic_regresion.pkl')
pipeline_one = joblib.load('pipeline_one.pkl')
pipeline_two = joblib.load('pipeline_two.pkl')


solutions = {
    "solution_1": {
        "model":{
            'name': 'model_logistic_regresion_001',
            'model': model_logistic_regresion_001
        },
        "pipeline":{
            'name':'pipeline_one',
            'pipeline': pipeline_one
        } 
    },
    "solution_2": {
        "model":{
            'name':'model_logistic_regresion_002',
            'model': model_logistic_regresion_002
        } ,
        "pipeline": {
            'name':'pipeline_two',
            'pipeline': pipeline_two
        }
    }
}

def predict(data):
    """
    Make predictions using the provided input data.

    Parameters
    ----------
    input_data : list
        List containing input data for prediction.

    Returns
    -------
    tuple
        Predicted credit risk and probabilities.
    """

    results = []

    for k,v in solutions.items():
        try:
            df = pd.DataFrame([data])
            item = v['pipeline']['pipeline'].transform(df)
            predict_proba = v['model']['model'].predict_proba(item).tolist()
            predict = v['model']['model'].predict(item).tolist()[0]
            resp = {
                'model' :v['model']['name'],
                'pipeline': v['pipeline']['name'],
                'predict': predict,
                'predict_proba': predict_proba,
                'status':'sucess'
            }
        except Exception as e:
            resp = {
                'model' :v['model']['name'],
                'pipeline': v['pipeline']['name'],
                'status': 'error',
                'error':str(e)
            }                  
        results.append(resp)
    print(results) 
    
    return results


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions, and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.
    """
    while True:

        msg = db.brpop(config.REDIS_QUEUE, config.SERVER_SLEEP)


        if msg is not None:
            # Convert string values to appropriate types
            msg = msg[1]
            newmsg = json.loads(msg)

            # Use the model to make predictions
            resp = predict(newmsg["data"])

            # Store the prediction results in a dictionary
            prediction_result = {
                "response": resp
            }

            job_id = newmsg["id"]
            # Store the results on Redis using the original job ID as the key
            db.set(job_id, json.dumps(prediction_result))

        # Sleep for a bit
        time.sleep(config.SERVER_SLEEP)


if __name__ == "__main__":
    print("Launching ML service...")
    classify_process()
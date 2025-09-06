import redis
import settings
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uuid
import json
import time


# Connect to Redis
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

app = FastAPI()


# Home page
@app.get("/")
def home():
    return {"message": "Welcome to the Loan Prediction API!"}

class CreditApplication(BaseModel):
    ID_CLIENT: int
    CLERK_TYPE: str
    PAYMENT_DAY: int
    APPLICATION_SUBMISSION_TYPE: str
    QUANT_ADDITIONAL_CARDS: int
    POSTAL_ADDRESS_TYPE: int
    SEX: str
    MARITAL_STATUS: int
    QUANT_DEPENDANTS: int
    EDUCATION_LEVEL: int
    STATE_OF_BIRTH: str
    CITY_OF_BIRTH: str
    NACIONALITY: int
    RESIDENCIAL_STATE: str
    RESIDENCIAL_CITY: str
    RESIDENCIAL_BOROUGH: str
    FLAG_RESIDENCIAL_PHONE: str
    RESIDENCIAL_PHONE_AREA_CODE: str
    RESIDENCE_TYPE: int
    MONTHS_IN_RESIDENCE: int
    FLAG_MOBILE_PHONE: str
    FLAG_EMAIL: int
    PERSONAL_MONTHLY_INCOME: float
    OTHER_INCOMES: float
    FLAG_VISA: int
    FLAG_MASTERCARD: int
    FLAG_DINERS: int
    FLAG_AMERICAN_EXPRESS: int
    FLAG_OTHER_CARDS: int
    QUANT_BANKING_ACCOUNTS: int
    QUANT_SPECIAL_BANKING_ACCOUNTS: int
    PERSONAL_ASSETS_VALUE: float
    QUANT_CARS: int
    COMPANY: str
    PROFESSIONAL_STATE: str
    PROFESSIONAL_CITY: str
    PROFESSIONAL_BOROUGH: str
    FLAG_PROFESSIONAL_PHONE: str
    PROFESSIONAL_PHONE_AREA_CODE: str
    MONTHS_IN_THE_JOB: int
    PROFESSION_CODE: float
    OCCUPATION_TYPE: float
    MATE_PROFESSION_CODE: float
    FLAG_HOME_ADDRESS_DOCUMENT: int
    FLAG_RG: int
    FLAG_CPF: int
    FLAG_INCOME_PROOF: int
    PRODUCT: int
    FLAG_ACSP_RECORD: str
    AGE: int
    RESIDENCIAL_ZIP_3: int
    PROFESSIONAL_ZIP_3: int

    # Add other necessary fields

@app.post("/prediction")
async def submit_credit_application(credit_application: CreditApplication):
    if not credit_application:
       raise HTTPException(status_code=400, detail="No data received")

    data = credit_application.dict()

    data_message = {"id": str(uuid.uuid4()), "data": data}

    job_data = json.dumps(data_message)

    job_id = data_message["id"]

    # Send the job to the model service using Redis
    db.lpush(settings.REDIS_QUEUE, job_data)

    # Wait for result model
    # Loop until we received the response from our ML model
    while True:
        # Attempt to get model predictions using job_id
        output = db.get(job_id)

        if output is not None:
            # Process the result and extract prediction and score
            output = json.loads(output.decode("utf-8"))

            #first_response_item = output["response"][0]

            #prediction = first_response_item["predict"]
            #probabilities = first_response_item["predict_proba"][0][0]
            #model = first_response_item["model"]
            #pipeline = first_response_item["pipeline"]

            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    # Determine the output message
    #if int(prediction) == 1:
    #    prediction = "Sorry Mr/Mrs/Ms , your loan is rejected!\n with a probability of {probabilities:.2f}%".format(probabilities=probabilities*100), model, pipeline
    #else:
    #    prediction = "Dear Mr/Mrs/Ms , your loan is approved! with a probability of {probabilities:.2f}%".format(probabilities=probabilities*100), model, pipeline

    return output
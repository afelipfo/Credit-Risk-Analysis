# Bora Bora: Machine Learning for Credit Risk 

## Table of Contents

- [Business Problem](#Business-Problem)
- [Deliverables](#Deliverables)
- [Dataset Access and Structure](#Dataset-Access-and-Structure)
- [Create Virtual Environment](#Create-Virtual-Environment)
- [Activate Docker](#Activate-Docker)
- [Workflow](#Workflow)

## Business Problem

**Industries:** *fintech*, *banks*, *logistics*, *delivery apps*, *insurtech*, *many others*

The fintech ecosystem has experienced rapid growth in recent years, establishing itself as a key actor in meeting the demands and needs of financial consumers. This growth was fueled by an increasing demand for financial services not provided by the traditional financial sector and by the increased demand for digital financial services due to the COVID-19 pandemic. In the US, companies leading these industries include [Stripe](https://stripe.com/docs/radar/risk-evaluation), [Affirm](https://www.affirm.com/business/blog/alternative-underwriting), [Brex](https://www.brex.com/), and [Marqeta](https://www.marqeta.com/platform/riskcontrol), among others. In the Latam region, the payments unit [Mercado Pago](https://www.mercadopago.com.ar/) from Mercado Libre and companies like [Nubank](https://nubank.com.br/en/), [Creditas](https://www.creditas.com/), [d-local](https://dlocal.com/), [Clip](https://clip.mx/), [Ualá](https://www.uala.com.ar/), [Clara](https://www.clara.com/), and more recently [Pomelo](https://pomelo.la/en/), are growing rapidly, constantly needing data analysis and modeling for problems such as credit risk analysis, fraud detection, customer churn prediction, or behavioral models to predict untimely payments.

Credit risk modeling is one of the most common uses of machine learning within this industry, aiming to use financial data to predict default risk. When a business or individual applies for a loan, the lender must evaluate whether the applicant can reliably repay the loan principal and interest. The machine learning model learns from data (such as firm information, financial statements, previous transactions, credit history, etc.) and can accurately predict the probability of repayment for a given business loan applicant.

This solution has broad applicability not only in fintech but also across many business sectors and industries such as logistics, banks, delivery apps, freight cargo companies, insurtech, etc., and could be easily adapted to other "risk estimation" business challenges.

## Deliverables

**Goal:** The main objective of this project was to build a service capable of predicting the credit scores of people based on financial transactional information. To properly test how this model would behave in a real environment, we ran a simulation of the model, showing each profile in the test dataset and determining whether it would assign credit to that person. We then evaluated whether a bank or fintech using this model would end up making more money or losing it.

**Main Deliverables:**

1. Exploratory Dataset Analysis (EDA) Jupyter notebooks and dataset
2. Scripts used for data pre-processing and data preparation
3. Training scripts and trained models, including a description of how to reproduce results
4. The model trained for credit score prediction
5. A simulation of your model making predictions on a testing dataset, documentation of the results and the simulation process
6. API with a basic UI interface for a demo (upload user's transactional data and return a score prediction)
7. Everything must be Dockerized and ready for deployment

## Dataset Access and Structure

The dataset for building this model(s) can be found in S3. We used the boto3 library to access the dataset.

- The dataset contains 54 variables in each field and 50,000 rows of training data.
- The dataset can be downloaded to your system and experimented with on your local development system given its size.

## Technical Aspects

In a nutshell, this project results in an API service that is backed by a machine learning model (or multiple models) that accepts a financial profile and predicts a credit risk score for that profile.

**Technologies and Tools:** *Supervised Learning*, *Deep Learning*, *HTTP APIs (FastAPI)*, *Scikit-learn*, *Pandas*, *Numpy*, *TensorFlow*, *Docker*, *Redis*, *Streamlit*.

For this project, we built three services:
  ```
├── api -> Used to implement the
communication interface between the users and our service

├── redis -> Used as a message broker, inside it has a task queue and a hash table.

├── model -> The code that implements the ML model, it receives jobs from Redis, processes them 
             with the model, and returns the results.
  ```
## Create Virtual Environment

1. Run the following command to create a virtual environment:

    ```
    python -m venv cra_env
    ```

2. Activate the virtual environment. Depending on your operating system, run:

    - On Windows:
        ```
        cra_env\Scripts\Activate
        ```

    - On Linux/Mac:
        ```
        source cra_env/bin/activate
        ```

3. Install project dependencies by executing the following command:

    ```
    pip install -r requirements.txt
    ```

With these steps, you will have created a virtual environment for your credit risk analysis project and installed the necessary dependencies.

## Activate Docker

This project has Dockerized the frontend application, backend, model resolution, and Redis.
To activate it, follow these steps:

1. If you are in the command line, navigate to ./creditriskanalysis and run:

    ```
    docker-compose up
    ```

If you are using Microsoft Visual Studio, you can right-click on the docker-compose.yml file and then select Docker-Compose up -Select Services. There you will have to choose which services to start.

## Workflow

Once you have tested your model and are ready to implement it, follow this process:

1. **File Creation:**
   - In the "src" folder, create a file with the naming convention "model_{TECHNIQUE_NAME}_{NUMBER}.py". For example, "model_logistic_regression_003.py".
   - Create a pipeline file with the naming convention "pipeline_{NUMBER}.py". Here, you will define your pipeline as a class.

2. **Model Definition:**
   - In the model file, import and call the class of the pipeline you created.
   - Train your pipeline using your training data. This may include steps such as preprocessing, feature engineering, and model selection.
   - Instantiate your model and train it using the pipeline defined above.
   - Add the necessary
   - 
3. **API  Integration:**
    -Develop an API endpoint that interacts with the trained model. In the  script app.py.
    -The API receives input data, validates it, and then sends it to the machine learning service for prediction.
    -The ML service processes this data, and the prediction result is sent back to the API, which then delivers it to the client.

4. **User Interface Development with Streamlit (streamlit.py)**
    -Develop a user-friendly interface using Streamlit, which allows users to input loan application data and submit it to the backend API for prediction.

# Bora Bora: Machine Learning for Credit Risk 

## ⚠️ IMPORTANTE: Configuración de Seguridad

**Antes de ejecutar este proyecto, lee el archivo [`README_SECURITY.md`](README_SECURITY.md) para configurar correctamente las credenciales de AWS.**

## Table of Contents

- [Business Problem](#Business-Problem)
- [Deliverables](#Deliverables)
- [Dataset Access and Structure](#Dataset-Access-and-Structure)
- [Security Configuration](#Security-Configuration)
- [Create Virtual Environment](#Create-Virtual-Environment)
- [Activate Docker](#Activate-Docker)
- [Workflow](#Workflow)

## Security Configuration

Este proyecto requiere credenciales de AWS para acceder a los datasets. **Por motivos de seguridad:**

1. **Configura las variables de entorno** siguiendo las instrucciones en [`README_SECURITY.md`](README_SECURITY.md)
2. **Nunca subas credenciales reales** a GitHub
3. **Usa el archivo `.env.example`** como plantilla para tu configuración local

## Business Problem

**Industries:** *fintech*, *banks*, *logistics*, *delivery apps*, *insurtech*, *many others*

The fintech ecosystem has experienced rapid growth in recent years, establishing itself as a key actor in meeting the demands and needs of financial consumers. This growth was fueled by an increasing demand for financial services not provided by the traditional financial sector and by the increased demand for digital financial services due to the COVID-19 pandemic. In the US, companies leading these industries include [Stripe](https://stripe.com/docs/radar/risk-evaluation), [Affirm](https://www.affirm.com/business/blog/alternative-underwriting), [Brex](https://www.brex.com/), and [Marqeta](https://www.marqeta.com/platform/riskcontrol), among others. In the Latam region, the payments unit [Mercado Pago](https://www.mercadopago.com.ar/) from Mercado Libre and companies like [Nubank](https://nubank.com.br/en/), [Creditas](https://www.creditas.com/), [d-local](https://dlocal.com/), [Clip](https://clip.mx/), [Ualá](https://www.uala.com.ar/), [Clara](https://www.clara.com/), and more recently [Pomelo](https://pomelo.la/en/), are growing rapidly, constantly needing data analysis and modeling for problems such as credit risk analysis, fraud detection, customer churn prediction, or behavioral models to predict untimely payments.

Credit risk modeling is one of the most common uses of machine learning within this industry, aiming to use financial data to predict default risk. When a business or individual applies for a loan, the lender must evaluate whether the applicant can reliably repay the loan principal and interest. The machine learning model learns from data (such as firm information, financial statements, previous transactions, credit history, etc.) and can accurately predict the probability of repayment for a given business loan applicant.

This solution has broad applicability not only in fintech but also across many business sectors and industries such as logistics, banks, delivery apps, freight cargo companies, insurtech, etc., and could be easily adapted to other "risk estimation" business challenges.

## Deliverables

**Goal:** The main objective of this project was to build a service capable of predicting the credit scores of people based on financial transactional information. To properly test how this model would behave in a real environment, we ran a simulation of the model, showing each profile in the test dataset and determining whether it would assign credit to that person. We then evaluated whether a bank or fintech using this model would end up making more money or losing it.

**Main Deliverables:**

1. Exploratory Dataset Analysis (EDA) Jupyter notebooks and dataset
2. Scripts used for data pre-processing and data preparation
3. Training scripts and trained models, including a description of how to reproduce results
4. The model trained for credit score prediction
5. A simulation of your model making predictions on a testing dataset, documentation of the results and the simulation process
6. API with a basic UI interface for a demo (upload user's transactional data and return a score prediction)
7. Everything must be Dockerized and ready for deployment

## Dataset Access and Structure

The dataset for building this model(s) can be found in S3. We used the boto3 library to access the dataset.

- The dataset contains 54 variables in each field and 50,000 rows of training data.
- The dataset can be downloaded to your system and experimented with on your local development system given its size.

## Technical Aspects

In a nutshell, this project results in an API service that is backed by a machine learning model (or multiple models) that accepts a financial profile and predicts a credit risk score for that profile.

**Technologies and Tools:** *Supervised Learning*, *Deep Learning*, *HTTP APIs (FastAPI)*, *Scikit-learn*, *Pandas*, *Numpy*, *TensorFlow*, *Docker*, *Redis*, *Streamlit*.

For this project, we built three services:
  ```
├── api -> Used to implement the
communication interface between the users and our service

├── redis -> Used as a message broker, inside it has a task queue and a hash table.

├── model -> The code that implements the ML model, it receives jobs from Redis, processes them 
             with the model, and returns the results.
  ```
## Create Virtual Environment

1. Run the following command to create a virtual environment:

    ```
    python -m venv cra_env
    ```

2. Activate the virtual environment. Depending on your operating system, run:

    - On Windows:
        ```
        cra_env\Scripts\Activate
        ```

    - On Linux/Mac:
        ```
        source cra_env/bin/activate
        ```

3. Install project dependencies by executing the following command:

    ```
    pip install -r requirements.txt
    ```

With these steps, you will have created a virtual environment for your credit risk analysis project and installed the necessary dependencies.

## Activate Docker

This project has Dockerized the frontend application, backend, model resolution, and Redis.
To activate it, follow these steps:

1. If you are in the command line, navigate to ./creditriskanalysis and run:

    ```
    docker-compose up
    ```

If you are using Microsoft Visual Studio, you can right-click on the docker-compose.yml file and then select Docker-Compose up -Select Services. There you will have to choose which services to start.

## Workflow

Once you have tested your model and are ready to implement it, follow this process:

1. **File Creation:**
   - In the "src" folder, create a file with the naming convention "model_{TECHNIQUE_NAME}_{NUMBER}.py". For example, "model_logistic_regression_003.py".
   - Create a pipeline file with the naming convention "pipeline_{NUMBER}.py". Here, you will define your pipeline as a class.

2. **Model Definition:**
   - In the model file, import and call the class of the pipeline you created.
   - Train your pipeline using your training data. This may include steps such as preprocessing, feature engineering, and model selection.
   - Instantiate your model and train it using the pipeline defined above.
   - Add the necessary
   - 
3. **API  Integration:**
    -Develop an API endpoint that interacts with the trained model. In the  script app.py.
    -The API receives input data, validates it, and then sends it to the machine learning service for prediction.
    -The ML service processes this data, and the prediction result is sent back to the API, which then delivers it to the client.

4. **User Interface Development with Streamlit (streamlit.py)**
    -Develop a user-friendly interface using Streamlit, which allows users to input loan application data and submit it to the backend API for prediction.


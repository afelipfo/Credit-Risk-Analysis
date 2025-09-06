FROM python:3

WORKDIR /model

COPY . .

RUN pip3 install -r requirements.txt

CMD ["python3", "ml_service.py"]
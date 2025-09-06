FROM python:3

WORKDIR /api

COPY . .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 5001

# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]
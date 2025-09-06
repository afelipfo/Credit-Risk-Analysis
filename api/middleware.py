from uuid import uuid4
import redis
import os
import json
import time

REDIS_IP = os.getenv("REDIS_IP", "redis")
REDIS_PORT = 6379
REDIS_DB_ID = 0
REDIS_QUEUE = "service_queue"
SERVER_SLEEP = 0.05

db = redis.Redis(host=REDIS_IP,
                       port=REDIS_PORT,
                       db=REDIS_DB_ID,
                    decode_responses=True) 

def model_predict():
    """
    Receives an image name and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    image_name : str
        Name for the image uploaded by the user.

    Returns
    -------
    prediction, score : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """

    # Assign an unique ID for this job and add it to the queue.
    # We need to assing this ID because we must be able to keep track
    # of this particular job across all the services
    # TODO
    job_id = str(uuid4())

    # Create a dict with the job data we will send through Redis having the
    # following shape:
    # {
    #    "id": str,
    #    "image_name": str,
    # }
    # TODO
    job_data = {
        'id': job_id,
        'image_name': f'{job_id}-text'
    }
    job_data = json.dumps(job_data)

    # Send the job to the model service using Redis
    # Hint: Using Redis `lpush()` function should be enough to accomplish this.
    # TODO
    db.lpush(REDIS_QUEUE, job_data)

    # Loop until we received the response from our ML model
    while True:
        # Attempt to get model predictions using job_id
        # Hint: Investigate how can we get a value using a key from Redis
        # TODO
        output =  db.get(job_id) 

        # Check if the text was correctly processed by our ML model
        # Don't modify the code below, it should work as expected
        if output is not None:
            output = json.loads(output)
            prediction = output["current_time"]
            score = output["current_time"]

            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(SERVER_SLEEP)

    return prediction, score
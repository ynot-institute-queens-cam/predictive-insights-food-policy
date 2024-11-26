"""
Finetune OpenAI GPT-3.5-turbo model
"""
import pathlib
from openai import OpenAI

# set OpenAI api key
OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

def create_file(filename: pathlib.Path, client):
    return client.files.create(file=open(filename, "rb"),  purpose="fine-tune").id

def create_fine_tune_job(client, train_file_id, validation_file_id):
    job = client.fine_tuning.jobs.create(
        training_file=train_file_id,
        validation_file=validation_file_id,
        model="gpt-3.5-turbo",
        hyperparameters={
        "n_epochs":2,
        "batch_size":1,
        "learning_rate_multiplier":2}
        )
    return job.id

def retrieve_job(client, job_id):
    return client.fine_tuning.jobs.retrieve(job_id).fine_tuned_model


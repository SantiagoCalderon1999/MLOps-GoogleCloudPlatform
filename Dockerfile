# Select an appropriate container image from: https://cloud.google.com/deep-learning-containers/docs/choosing-container
#FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-8

FROM python:3.9.5-slim

WORKDIR /

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# Copies the trainer code to the docker image.
COPY trainer /trainer
COPY scripts /scripts

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]


# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN apt update
RUN apt install build-essential -y
RUN apt-get update 
RUN apt-get install manpages-dev -y
RUN apt-get install git -y

RUN pip install --upgrade pip

# Install production dependencies.
RUN pip install -r requirements.txt


# RUN pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
RUN pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
# RUN pip3 install pycocotools==2.0.0

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 0 main:app
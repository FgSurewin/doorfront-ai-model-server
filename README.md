# DoorFront Detection Model

There are two ways to setup the development environment of DoorFront detection model on you local machine. 
* Python virtural environment
* Docker image

Pick the one that you find convenient.

## 1. Python Virtual Environment
To avoid disrupting your original python package, I suggest we should create a brand new virtual environment for this project. Follow the instructions below to create the virtual environment.

**Beofore you start, please make sure you have python on your local machine.**

### 1.1. Create python virtual environment
```bash
python -m venv ModelVenv
```

### 1.2. Activate python virtual environment
```bash
source ./ModelVenv/Scripts/activate
```

### 1.3. Upgrade `pip` of the new virtual environment
```bash
python -m pip install --upgrade pip
```

### 1.4. Install the required packages according to the `requirements.txt` file
```bash
pip install -r ./requirements.txt
```

### 1.5 Install `pycocotools` manually 
```bash
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

### 1.6 Run model
```bash
python main.py
```

Now, you can test the model through webpage `localhost:5000`
(Copy this link to any browser and hit enter)

## 2. Docker Image

I recommend you use this method to run the model, although you don't know docker before. 

### 2.1 Install `Docker`
[Docker Home Page](https://www.docker.com/get-started/)

Download `Docker Destop` and install it on your local machine. 


### 2.2 Switch to development mode
Go to the `Dockerfile`
```Dockerfile
# Uncomment this line since we are running in development mode
CMD python main.py



# Comment this line since we are running in development mode
# CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 0 main:app

```

### 2.3 Build docker image
```bash
docker build -t doorfront-ai-model .
```



### 2.4 Run model
```bash
docker run -it -p 5000:5000 --name doorfront-model doorfront-ai-model
```
Now, you can test the model through webpage `localhost:5000`
(Copy this link to any browser and hit enter)

### End.
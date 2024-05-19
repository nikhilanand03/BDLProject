# BigDataLabProject

## 1. Preprocessing data

The first step is to preprocess the data. You can fetch the data files and preprocess them by following these instructions:

- Clone the repository to a directory named "BigDataLabProject".

```
export AIRFLOW_HOME=~/BigDataLabProject
source ~/airflow_venv/bin/activate
sudo kill <PID>
airflow db init

airflow scheduler \
--pid ${AIRFLOW_HOME}/logs/airflow-scheduler.pid \
--stdout ${AIRFLOW_HOME}/logs/airflow-scheduler.out \
--stderr ${AIRFLOW_HOME}/logs/airflow-scheduler.out \
-l ${AIRFLOW_HOME}/logs/airflow-scheduler.log \
 -D
```

- Then open a new terminal and run the following:

```
export AIRFLOW_HOME=~/airflow 
source ~/airflow_venv/bin/activate 
airflow users create -e <EMAILID> -f <FIRSTNAME> -l <LASTNAME> -p <PASSWORD> -u <USERNAME> -r Admin
airflow webserver -p 8080
```

- Now in the webserver, find the dag named "keypoints_dag". Run this dag and you will find the required files in the data/ directory.

## 2. Experimentation of Model Hyperparameters (MLFlow)

In the `mlflow_part` directory, the notebook `final_colab_notebook_mlflow.ipynb` has all the code which can be run on Google Colab with a T4 GPU to recreate the training results of this project. The final model we used was the one obtained in the second experiment. We obtained a validation loss of 11.15 (it reduced from 52 to 18.15 over the course of 10 training epochs).


| Learning rate | Epochs | Final validation loss |
| -------- | ------- | ------- |
| 8e-6 | 10 | 44.001 |
| 8e-5 | 10 | 18.15 |
| 8e-5 (7) + 4e-5 (3) | 7 + 3 | 24.01 |
| 8e-5 (12) + 4e-5 (8) | 12 + 8 | 11.26 |
| 8e-5 (12) + 4e-5 (11) + 8e-6 (2) | 12 + 11 + 2 | 10.97 |
# project_disaster_response_pipeline
Project along the Udacity Nanodegree for Data Science using pipelines. The project is about extracting valuable information out of social media messages in case of disasters. First a ETL pipeline is created to clean the data, next a Machine Learning pipeline is created extract information out of the data and finally a web-app is created to access the results.

The github repository can be found here: https://github.com/mmfill/electric-motor.git

The blog can be found here: https://medium.com/@matthias.fill/how-to-improve-the-electric-car-250cd92f2793

## Short description:

Looking at the data from Paderborn Universität to explain motor temperature and torque of an electric motor by other parameters.

Needed packages in Python3:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

files in repository:

electric-motor.ipynb Jupyter notebook
## Motivation:

Personal: I had to to a project for the Udacity Data Science Nanodegree. I chose the dataset of the electric motor, because the topic interests me, but I did not know a lot about. General: Internal parameters like temperature and torque are important features to optimize the efficiency of an electric motor. But they are hard to measure in a driving vehicle. Can we predict both features with "outside" and easy-to-measure parameters like outside temperature, current or voltage? For this data was collected at 2Hz for different runs which lasted 1-6 hours.

Results: All results were achieved with a simple linear regression model. Current i_q is ideal to predict torque. This feature alone explains 99.36% of the variance of torque with a simple linear regression model. This is true when looking at the data of a single run. Motor temperature is more difficult to predict and the given parameters explain about 58% of the variance of motor temperature. This is true when looking at the data of a single run. When combining the data of different runs the explained variance of motor temperature drops to 34% while the explained variance of torque drops to 99.32%.

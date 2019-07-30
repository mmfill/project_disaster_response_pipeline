# project_disaster_response_pipeline

## Short description:
Project along the Udacity Nanodegree for Data Science using pipelines. The project is about extracting valuable information out of social media messages in case of disasters. First a ETL pipeline is created to clean the data, next a Machine Learning pipeline is created extract information out of the data and finally a web-app is created to access the results.

The github repository can be found here: https://github.com/mmfill/project_disaster_response_pipeline

## Needed packages in Python3:
* import pandas as pd
* from sqlalchemy import create_engine

## EPL Pipeline:
The EPL Pipeline first extracts information out of two .csv files and writes them into two dataframes. Both dataframes get merged and cleaned. Most importantly the 'categories' column gets expanded to individual categories with integer entries 0 and 1.

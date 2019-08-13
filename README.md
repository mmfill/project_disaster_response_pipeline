# project_disaster_response_pipeline

## Short description:
Project of the Udacity Nanodegree for Data Science using pipelines. The project is about extracting valuable information out of social media messages in case of disasters. First the data is merged and cleaned before it is stored in a database. Next a Machine Learning pipeline is created to train a model on the data. Finally a web-app is created to categorise input messages and to visualize origins of messages and origin of individual categories.

The github repository can be found here: https://github.com/mmfill/project_disaster_response_pipeline

## How to run it:
### Data wrangling
The file data/process_data.py loads two csv files into a dataframe, merges the dataframe and cleanes the data. The data is finally stored in a dataframe. The file can be run via the command `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`.

### Machine learning
The file model/train_classifier.py loads the data from the database and trains a Classifier (RandomForrest) onto the data. The goal is to classify messages along 36 categories in case of disaster. The model is classified and the trained model is stored as a Pickle file. The pyhton file can be run with the command 'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'

### Web app
To run the web app type `python app/run.py`. It shows the distribution of categories along genres and distribution of message genres of the training data in the two csv files. Input messages are catergorised. To load the webapp go to http://0.0.0.0:3001/

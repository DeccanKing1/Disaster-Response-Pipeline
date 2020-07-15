# Disaster-Response-Project:
Using ETL pipeline and NLP pipeline as part of disaster response to solve multiclass multioutput problem which will classify the message sent during disaster to 
appropriate department and the deploying solution to webapp including data visualisation.

## Project Overview :
In this project, we analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. A data set containing real messages that were sent during disaster events is in messages dataset. We will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.  Project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. Once we get classification that department will get notified to that message so that proper help can be sent there immidiately. 


## Steps to run the project:
1. Run process_data.py to export data to a database file (will have to mention location of message data)
To run ETL pipeline that cleans the data and store in database: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
2. Run train.py to export machine learning model to pickle file (give location of database file from process_data.py ETL output)
- To run ML pipeline that trains classifier and saves:  `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run run.py to redirect to webapplication
  - point location of pickle file from train.py output
  - Open separate terminal window and run env|grep WORK
  - From the output you get replace the SPACEID and SPACEDOMAIN field in the following URL and you can access the webapp https://SPACEID-3001.SPACEDOMAIN
- Run the following command in the app's directory to run your web app:  `python run.py`
   

## Overview of the project

This project required 3 steps:
  1. Create ETL of data from CSV files and upload cleansed data to a database
  2. Write machine learning algorithm to analyze messages and optomize model to correctly classify labels for that text
  3. Create web aplication that can show graphs of overviews of the messages data, as well as a text bar that could read a message and correctly classify the message to corrct classes

## Data Files in repository
#### 1. Categories of messages (disaster_categories.csv): 
CSV of categories of all messages (36 possible categories)

#### 2. Messages (disaster_messages.csv):
CSV file of all disaster messages

#### 3. Database file (DisasterResponse.db):
Database file that is the output of the ETL section
  
  
## Program Files in the Repository

### ETL Section (process_data.py)

The ETL secition is your typical extract - transform - load process. Data was provided in CSV's that needed to be read into pandas dataframes, merged together, and then cleaned. 
That pandas Dataframe was then loaded to a SQLite database (hosted by Udacity) to be loaded by the next step of the project.

### Machine Lerning Pipelines (train.py)
Problem is multiclass multioutput supervised machine learning problem
After loading the data into a pandas dataframe, data was broken up into the feature columns and the response columns, and appropriate features and grissearch parameters were added to the function.

### Web Application (run.py)

The web application shows 3 different bar charts: Genre frequency, top 5 response features, and the top 5 words most frequently used in the messages.

### References
Figure Eight for providing the relevant dataset to train the model: https://appen.com/

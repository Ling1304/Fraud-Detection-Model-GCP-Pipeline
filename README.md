# Fraud Detection Model with GCP Pipeline
This project aims to build a decent fraud detection model using a Kaggle dataset, employing advanced machine learning techiniques. The end-to-end pipeline for data processing, model training, and deployment is developed on Google Cloud Platform (GCP) with a Dockerized workflow, making it scalable and easy to manage. The focus of this project is mainly on developing a simple end-to-end GCP Pipeline that automates every stage of the machine learning workflow, from data ingestion and preprocessing to model training, evaluation, and deployment. 

## Project Overview:
Data Acquisition: The dataset is sourced from Kaggle, and can be found here: https://www.kaggle.com/competitions/ieee-fraud-detection/overview
Data Cleaning & Preprocessing: Using preprocess.py, the dataset is cleaned and preprocessed for model training. 
Model Training: The train.py script trains a fraud detection model that uses various features from the dataset to achieve a respectable accuracy in detecting fraudulent activity.
Dockerization: The train.py and preprocess.py scripts are containerized with Docker for streamlined deployment. A Dockerfile is included, which includes all dependencies to ensure the scripts run across environments.
GCP Pipeline: The Dockerized code is pushed to Google Cloud Artifacts and executed in a GCP pipeline. This pipeline automates the entire process, from data ingestion and model training to saving artifacts.
Model and Metrics Storage: The trained model is stored in a specified GCP bucket, and evaluation metrics (AUC scores, cross-validation metrics, etc.) are saved into a JSON file for tracking model performance over time.

## Repository Structure:
Dockerfile: Defines the container configuration for the project.
preprocess.py: Script for data cleaning and preprocessing.
train.py: Script for training the fraud detection model.
requirements.txt: Text file that includes all the dependencies needed. 

## Future Updates:
This repository will be updated periodically to enhance the pipeline's workflow. Future updates may include the integration of automated data ingestion steps, as well as hyperparameter tuning to optimize the model's performance. Additionally, improvements in model evaluation, real-time prediction features, and more advanced fraud detection algorithms will be considered to further improve the model’s accuracy. These updates aim to ensure the pipeline remains efficient, scalable, and adaptive to new data and fraud detection techniques.

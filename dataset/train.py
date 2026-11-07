# step1:import all necessary libraries
import pandas as pd   #used to read the csv file and manipulate tabular data data
import numpy as np # used for numerical operations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#To measure the model performance 
from sklearn.metrics import (mean_squared_error, r2_score,mean_absolute_error)

#path helps locate the files
from pathlib import Path

#joblib is used to save the trained model and feature names 
import joblib

#step2:Read the csv dataset
data= Path("dataset/16_air_quality_prediction.csv")
df=pd.read_csv(data)
print("Dataset loaded successfully!")

#step3:Explore the dataset(EDA)

#find num of rows and columns
print("\nDataset shape:")
print(df.shape)

#Display dataset information
print("\nDataset Information:")
print(df.info())

#Display first 5 rows of the dataset
print("\nFirst five rows:")
print(df.head())

#Display statistical information
print("\nStatistical Information:")
print(df.describe())

#check missing values
print("\nMissing Values:")
print(df.isnull().sum())

#check duplicate values
print("\nDuplicate Values:")
print(df.duplicated().sum())

#step4:Prepare the data

#convert date column to datetime format
df["date"]=pd.to_datetime(df["date"],errors="coerce") 

#Extract year, month, day
df["year"]=df["date"].dt.year
df["month"]=df["date"].dt.month 
df["day"]=df["date"].dt.day


#Remove rows with missing values
df = df.dropna(
    subset=["year", "month", "day", "location", "pm2.5", "pm10", "temperature", "predicted_aqi"])

#convert location into numbers
df=pd.get_dummies(df, columns=["location"], dtype=int)

#seperate features and target
x=df.drop(["date","predicted_aqi"],axis=1)
y=df["predicted_aqi"]

#step5:Split the data 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


#Train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)
print("\nModel trained successfully!")

#step6:Evaluate the model
y_test_pred = model.predict(x_test)

#mean absolute error
mae = mean_absolute_error(y_test, y_test_pred)

#Root mean squared error
rsme = np.sqrt(mean_squared_error(y_test, y_test_pred))

#R2 score
r2 = r2_score(y_test, y_test_pred)


#Display the results
print("\nAQI Model Evaluation Results")
print("----------------------------")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rsme:.2f}")
print(f"R2 Score: {r2:.2f}")

#save the trained model
joblib.dump(model, "aqi_model.joblib")

joblib.dump(
    list(x.columns),
    "aqi_feature_names.joblib"
)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:23:51 2021

@author: konam
"""

# Importing the necessary libraries for the project
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import streamlit as st

st.title("Predicting a Student Score")
url = "http://bit.ly/w-data"
# Function for getting the data
def get_data(url):
    data = pd.read_csv(url)
    return (data)

# Function for describing the data
def about_data(df):
    col1,col2,col3 = st.beta_columns(3)
    with col1:
        st.write("About the task")
        st.write("Predict the percentage of a student based on the no. of study hours.")
    with col2:
        st.write("Given Data")
        st.write(df)
    with col3:
        st.write("Data Statistics")
        st.write(df.describe())

# Function for data splitting
def data_split(df):
    x = np.array(df.Hours)
    y = np.array(df.Scores)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state=0)
    x_train = x_train.reshape(-1,1)
    x_test = x_test.reshape(-1,1)
    return (x,y,x_train,x_test,y_train,y_test)

# Function for building model
def model(x_train,y_train):
    regr = LinearRegression()
    regr.fit(x_train, y_train)
    return(regr)

#Reading the Functions
s_data = get_data(url)
about_data(s_data)
x,y,x_train,x_test,y_train,y_test = data_split(s_data)
regr = model(x_train,y_train)

#Comparing the Results and plotting the chart
st.write(f"Regression line: Scores = {regr.coef_[0].round(2)} * Hours + {regr.intercept_.round(2)}")
y_pred = regr.predict(x_test)

col1,col2 = st.beta_columns(2)
with col1:
    comp = pd.DataFrame({"Actual":y_test,"Prediction":y_pred.round(0)})
    st.write("Comparing the Actual Vs Predicted values",comp)
    st.write(f"R2 score of the regression model is: {metrics.r2_score(y_pred, y_test).round(2)}")
    st.write(f"Mean_absolute_error of the regression model is: {metrics.mean_absolute_error(y_pred,y_test).round(2)}")
with col2:
    st.write("Plotting the data points and the regression line")
    line = regr.coef_ * x + regr.intercept_
    fig,ax = plt.subplots()
    ax.scatter(x,y,color = "blue")
    ax.plot(x, line, color = 'green')
    ax.set_xlabel("Hours")
    ax.set_ylabel("Scores")
    st.pyplot(fig)

#Checking the target with user input and printing the score
st.sidebar.write("Enter study hours to predict the scores")
option = st.sidebar.text_input('Study Hours',value = 9.5)
max_study_hours = (100-regr.intercept_)/(regr.coef_)

def pred(hours):
    predicted_score = regr.predict(hours)
    st.sidebar.write(f"Predicted Score: {predicted_score[0].round(2)} for the study hours:{p_input}")

p_input = eval(option)
if(p_input >= max_study_hours):
    st.sidebar.write("Maximum Study hours reached")
elif(p_input<0):
    st.sidebar.write("Please enter a positive value")
else:
    hours = [[p_input]]
    pred(hours)

st.sidebar.write(f"Note1: The predicted score attains 100 when the study hours is {max_study_hours[0].round(2)}, so request to enter the hours less than the max hours.")
st.sidebar.write("Note2: The model predicts minimum score even when the hours is 0 ,since the model was build with less data and no scores with 0 hours on training data.")
#print(metrics.mean_absolute_error(y_test,y_pred))

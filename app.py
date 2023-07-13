import uvicorn 
from fastapi import FastAPI
from depression import Depression
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import random
from pydantic import BaseModel
from typing import Union

gad7_levels = {
        "Minimal Anxiety": (0,4),
        "Mild Anxiety": (5,9),
        "Moderate Anxiety": (10, 14),
        "Severe Anxiety": (15,35)
    }



class Anxiety(BaseModel):
    D1: int
    D2: int
    D3: int
    D4: int
    D5: int
    D6: int
    D7: int
    

app = FastAPI()
pickle_in = open("Anxiety.pkl","rb")
model = pickle.load(pickle_in)

# def get_severity(prediction: int) ->str:
#     for level, (lower, upper) in gad7_levels.items():
#         if lower <= prediction <= upper:
#             return level
#     return "unknown"

# def get_message(severity_level: str) -> tuple[str, str]:
#     if severity_level == "Minimal Anxiety":
#         message = "Your anxiety level is minimal."
#         advice = "Continue practicing self-care and maintaining a healthy lifestyle."
#     elif severity_level == "Mild Anxiety":
#         message = "You are experiencing mild anxiety."
#         advice = "Consider incorporating stress management techniques into your daily routine."
#     elif severity_level == "Moderate Anxiety":
#         message = "Your anxiety level is moderate."
#         advice = "Seek support from a healthcare professional or therapist for further guidance."
#     elif severity_level == "Severe Anxiety":
#         message = "You are experiencing severe anxiety."
#         advice = "It is recommended to consult with a mental health professional for proper assessment and treatment."
#     else:
#         message = "Unable to determine the severity level."
#         advice = "Please consult with a healthcare professional for further evaluation."
    
#     return message, advice


@app.get("/")
def read_root():
    return{"hello": "world"}

@app.post("/anxiety")
def predict_anxiety(data:Anxiety):
    data=data.dict()
    D1 = data['D1']
    D2 = data['D2']
    D3 = data['D3']
    D4 = data['D4']
    D5 = data['D5']
    D6 = data['D6']
    D7 = data['D7']

    
    prediction = model.predict([[D1, D2, D3, D4, D5,D6,D7]])

    severity = "unknown"
    message = ""
    advice = ""

    if prediction >= gad7_levels["Minimal Anxiety"][0] and prediction <= gad7_levels["Minimal Anxiety"][1]:
        severity = "Minimal Anxiety"
        message = "Your anxiety level is minimal."
        advice = "Continue practicing self-care and maintaining a healthy lifestyle."
    elif prediction >= gad7_levels["Mild Anxiety"][0] and prediction <= gad7_levels["Mild Anxiety"][1]:
        severity = "Mild Anxiety"
        message = "You are experiencing mild anxiety."
        advice = "Consider incorporating stress management techniques into your daily routine."
    elif prediction >= gad7_levels["Moderate Anxiety"][0] and prediction <= gad7_levels["Moderate Anxiety"][1]:
        severity = "Moderate Anxiety"
        message = "Your anxiety level is moderate."
        advice = "Seek support from a healthcare professional or therapist for further guidance."
    elif prediction >= gad7_levels["Severe Anxiety"][0] and prediction <= gad7_levels["Severe Anxiety"][1]:
        severity = "Severe Anxiety"
        message = "You are experiencing severe anxiety."
        advice = "It is recommended to consult with a mental health professional for proper assessment and treatment."


    return {
        # "total_score": prediction,
        "severity_level": severity,
        "message": message,
        "advice": advice
    }




if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


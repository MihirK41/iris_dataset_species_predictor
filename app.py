import pickle
from flask import Flask, render_template, request
import numpy as np 
import pandas as pd

model = pickle.load(open('iris.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")



@app.route('/predict', methods=["POST"])
def predict_species():
    if request.method == "POST":
        sepal_length = float(request.form['sl'])
        sepal_width = float(request.form['sw'])
        petal_length = float(request.form['pl'])
        petal_width = float(request.form['pw'])
        result = model.predict(np.array([sepal_length,sepal_width,petal_length,petal_width]).reshape(1,4))
        
        if result[0]=="setosa":
            result = "setosa"
        elif result[0]=="versicolor":
            result = "versicolor"
        else:
            result = "virginica"
        return result
    
if __name__=="__main__":
    app.run(debug=True)
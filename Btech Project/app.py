from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('usedvehicle2.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data3 = int(data3)
    data4 = request.form['d']
    data4 = int(data4)
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data8 = float(data8)
    data9 = request.form['i']
    data9 = float(data9)
    data10 = request.form['j']
    data10 = float(data10)
    arr=pd.DataFrame([[data1, data2, data3, data4 ,data5, data6, data7, data8,data9, data10]], columns=['Name','Location','Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Power','Seats'])
    # arr = pd.DataFrame([[data1, data2, data3, data4 ,data5, data6, data7, data8,data9, data10]])
    pred = model.predict(arr)
    print(pred)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















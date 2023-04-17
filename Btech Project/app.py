from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('usedvehicle2.pkl', 'rb'))
model1 = pickle.load(open('loan.pkl', 'rb'))
app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')

@app.route('/loan')
def man2():
    return render_template('loan_home.html')

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
    return render_template('after.html', data=pred)

@app.route('/Loanpredict', methods=['POST'])
def loan():
    data1 = request.form['a']
    data1 = int(data1)
    data2 = request.form['b']
    data2 = int(data2)
    data3 = request.form['c']
    data3 = float(data3)
    data4 = request.form['d']
    data4 = int(data4)
    data5 = request.form['e']
    data5 = int(data5)
    data6 = request.form['f']
    data6 = int(data6)
    data7 = request.form['g']
    data7 = int(data7)
    data8 = request.form['h']
    data8 = float(data8)
    data9 = request.form['i']
    data9 = int(data9)
    data10 = request.form['j']
    data10 = int(data10)
    data11 = request.form['k']
    data11 = int(data11)
    data12 = request.form['l']
    data12 = int(data12)
    arr=pd.DataFrame([[data1, data2, data3, data4 ,data5, data6, data7, data8,data9, data10,data11,data12]], columns=['loan_amount','asset_cost','loan_to_asset_value_ratio','asset_manufacturer_id','credit_score','new_loan_accounts_in_last_6_months','overdue_accounts_in_last_6_months','avg_account_age','active_loan_accounts','existing_loan_balance','total_disbursed_amount','current_installment'])
    pred = model1.predict(arr)
    return render_template('loanafter.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















from flask import Flask, render_template, redirect, request, url_for, session, flash, send_file
from datetime import datetime
import xlsxwriter

app2 = Flask(__name__)
app2.secret_key = "abc"
import requests, json

city = "Baghdad"

key = '51d9c32602164726b7482248222708'

@app2.route("/")
def home_page():
    current = get_day()
    print(current)
    return render_template("index.html",data=current,current=current,city=city)

@app2.route("/day")
def day():
    data = get_day1()
    return render_template("day.html",data=data,city=city)

@app2.route("/72")
def h72():
    data = get_day3()
    return render_template("72.html",data=data,city=city)

@app2.route("/7day")
def days7():
    data = get_day7()
    return render_template("7days.html",data=data,city=city)


@app2.route('/post',methods = ['POST'])
def login():
    global city
    city = request.form['city']
    return redirect(url_for('home_page'))


@app2.route('/post1',methods = ['POST'])
def login1():
    global city
    city = request.form['city']
    return redirect(url_for('day'))



@app2.route('/post2',methods = ['POST'])
def login2():
    global city
    city = request.form['city']
    return redirect(url_for('days7'))



@app2.route('/post3',methods = ['POST'])
def login3():
    global city
    city = request.form['city']
    return redirect(url_for('home_page'))


@app2.route("/download")
def downloadFile ():
    datef = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    path = str(datef)+".xlsx"
    create_file(path)
    return send_file(path, as_attachment=True)


def get_day():
    BASE_URL = 'https://api.weatherapi.com/v1/forecast.json?key='+key+'&q='+city
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        data = response.json()
        current = data['current']
        return current
    else:
        print("Error in the HTTP request")



def get_day1():
    BASE_URL = 'https://api.weatherapi.com/v1/forecast.json?key='+key+'&q='+city
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        data = response.json()
        data = data['forecast']
        data = data['forecastday'][0]
        date = data['hour']
        return date
    else:
        print("Error in the HTTP request")

def get_day30():
    BASE_URL = 'https://api.weatherapi.com/v1/forecast.json?key='+key+'&q='+city+'&days=30'
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        data = response.json()
        data = data['forecast']
        data = data['forecastday']
        return data
    else:
        print("Error in the HTTP request")

def create_file(path):
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet()
    worksheet.set_column('A:A', 20)
    bold = workbook.add_format({'bold': True})
    data = get_day1()
    for dd in data:
            worksheet.write('A1', 'last updated')
            worksheet.write('B1', dd["time"])
            worksheet.write('A2', 'humidity')
            worksheet.write('B2', dd["humidity"])
            worksheet.write('A3', 'DD')
            worksheet.write('B3', dd["wind_degree"])
            worksheet.write('A4', 'FF')
            worksheet.write('B4', dd["wind_mph"])
            worksheet.write('A5', 'P0P0P0 hpa')
            worksheet.write('B5', dd["pressure_mb"])
            worksheet.write('A6', 'VV')
            worksheet.write('B6', dd["vis_km"])
            worksheet.write('A7', 'TTT')
            worksheet.write('B7', dd["temp_c"])
            worksheet.write('A8', 'TDTDTD c')
            worksheet.write('B8', dd["dewpoint_c"])
            worksheet.write('A9', 'UV')
            worksheet.write('B9', dd["uv"])
    workbook.close()



def get_day3():
    BASE_URL = 'https://api.weatherapi.com/v1/forecast.json?key='+key+'&q='+city+'&days=3'
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        data = response.json()
        data = data['forecast']
        data = data['forecastday']
        return data
    else:
        print("Error in the HTTP request")

def get_day7():
    BASE_URL = 'https://api.weatherapi.com/v1/forecast.json?key='+key+'&q='+city+'&days=7'
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        data = response.json()
        data = data['forecast']
        data = data['forecastday']
        return data
    else:
        print("Error in the HTTP request")





#https://api.weatherapi.com/v1/forecast.json?key=91bfb0102d8b43c3adb155104221108&q=07112&days=3


if __name__=="__main__":
    app2.run(debug=True,port=5002)

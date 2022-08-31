# importing modules
import requests, json

#BASE_URL = 'https://api.weatherapi.com/v1/forecast.json?key=91bfb0102d8b43c3adb155104221108&q=07112&days=1'
BASE_URL = 'https://api.weatherapi.com/v1/forecast.json?key=91bfb0102d8b43c3adb155104221108&q=07112&days=4'
response = requests.get(BASE_URL)
if response.status_code == 200:
    data = response.json()
    data = data['forecast']
    data = data['forecastday']
    for dd in data:
        d = dd['hour']
        print(d)


else:
    print("Error in the HTTP request")
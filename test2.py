import requests, json

#IBM KEY  :  41b54a6111754445b54a611175b44574

URL = "https://api.weather.com/v3/wx/forecast/daily/7day?geocode=33.74,-84.39&format=json&units=e&language=en-US&apiKey=41b54a6111754445b54a611175b44574"

#91bfb0102d8b43c3adb155104221108
##

response = requests.get('https://api.weatherapi.com/v1/forecast.json?key=91bfb0102d8b43c3adb155104221108&q=07112&days=1')
print(response.text)



#http://api.weatherapi.com/v1/forecast.xml?key=91bfb0102d8b43c3adb155104221108&q=07112&days=7
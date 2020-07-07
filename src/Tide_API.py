#importing required packages
import requests
import re
from datetime import datetime
import json


#converting datetime object to unix time
def unix_time(dt):
    epoch = datetime.utcfromtimestamp(0)
    return ((dt - epoch).total_seconds())

#hitting the API and returning the tide on the given day
def get_tide(coords,dt):
    min_longitude,min_latitude,max_longitude,max_latitude = coords
    lat = str((min_latitude+max_latitude)/2)
    lon = str((min_longitude+max_longitude)/2)
    base_url = 'https://www.worldtides.info/api?heights'
    ut = str(unix_time(dt))
    #generating the url
    query_string = '&lat={lat}&lon={lon}&start={ut}&length=1000&key={api_key}'\
    .format(lat = lat,lon = lon,ut = ut, api_key = get_API_key())
    url = base_url + query_string
    #hitting API
    r = requests.get(url)
    #converting respoiinse to JSON
    tide = json.loads(r.text)
    return tide['heights'][0]['height']

def get_API_key():
    params_f = open('config/data-params.json')
    params = json.load(params_f)
    return params['world_tide_API_key']

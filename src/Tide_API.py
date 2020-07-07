import requests
import re
from datetime import datetime
import json


def unix_time(dt):
    """
    Params - 1. dt (datetime) - date that needs to be converted to unix time
    Return - int - unix time in seconds
    """
    epoch = datetime.utcfromtimestamp(0)
    return ((dt - epoch).total_seconds())


def get_tide(coords,dt):
    """
    Params - 1. coords ([min-x, min-y, max-x, max-y]) - coral reef bounding box
             2. dt (datetime) - date we want the tide on
    Return - int - tide level on given day
    """
    #get average lat and lon
    min_longitude,min_latitude,max_longitude,max_latitude = coords
    lat = str((min_latitude+max_latitude)/2)
    lon = str((min_longitude+max_longitude)/2)
    #generating the url of API to hit
    base_url = 'https://www.worldtides.info/api?heights'
    ut = str(unix_time(dt))
    query_string = '&lat={lat}&lon={lon}&start={ut}&length=1000&key={api_key}'\
                        .format(lat = lat,lon = lon,ut = ut, api_key = get_API_key())
    url = base_url + query_string
    #hitting API and storing contents in json format
    r = requests.get(url)
    tide = json.loads(r.text)
    return tide['heights'][0]['height']

def get_API_key():
    """
    Loading in API ket from data params
    Return - str - API key
    """
    #loads in data params and returns API key
    params_f = open('config/data-params.json')
    params = json.load(params_f)
    return params['world_tide_API_key']

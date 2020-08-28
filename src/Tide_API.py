import requests
import re
from datetime import datetime
import json
from datetime import timedelta
import src.Time_Zone_API as tz_api

def unix_time(dt):
    """
    Params - 1. dt (datetime) - date that needs to be converted to unix time
    Return - int - unix time in seconds
    """
    # dt = dt + timedelta(hours = 12)
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
    lat = (min_latitude+max_latitude)/2
    lon = (min_longitude+max_longitude)/2
    while lon < -180:
        lon += 360
    while lon > 180:
        lon -= 180
    #get API key
    tide_API_key = get_API_key()
    #getting local time
    ut = unix_time(dt)

    time_diff = tz_api.get_offset(coords,ut)

    local_time = str(ut + time_diff)
    #generating the url of API to hit
    base_url = 'https://www.worldtides.info/api/v2?heights'
    query_string = '&lat={lat}&lon={lon}&start={ut}&length=1000&key={api_key}'\
                        .format(lat = str(lat),lon = str(lon),ut = local_time, api_key = tide_API_key)
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
    tide_API_key = params['world_tide_API_key']
    return tide_API_key

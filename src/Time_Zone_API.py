import requests
import json



def get_offset(coords,ut):
    """
    Params - 1. coords ([min-x, min-y, max-x, max-y]) - coral reef bounding box
             2. ut (int) - unix time stamp of date
    Return - int - timezone offset from utc
    """
    #get average lat and lon
    min_longitude,min_latitude,max_longitude,max_latitude = coords
    lat = (min_latitude+max_latitude)/2
    lon = (min_longitude+max_longitude)/2
    while lon < -180:
        lon += 360
    while lon > 180:
        lon -= 180
    API_key = get_API_key()
    ut = int(ut)
    #generating the url of API to hit
    base_url = 'http://api.timezonedb.com/v2.1/get-time-zone'
    query_string = '?key={key}&format=json&by=position&lat={lat}&lng={lng}&time={ut}'\
                        .format(key = API_key, lat = lat, lng = lon, ut = ut)
    url = base_url + query_string

    #hitting API and storing contents in json format
    r = requests.get(url)

    timezone = json.loads(r.text)
    print(timezone)
    return timezone['gmtOffset']


def get_API_key():
    """
    Loading in API ket from data params
    Return - str - API key
    """
    #loads in data params and returns API key
    params_f = open('config/data-params.json')
    params = json.load(params_f)
    return params['tz_API_key']

#importing required packages
import requests
import re
from datetime import datetime
import json
import os

#storing API Key
os.environ['API_KEY'] = 'c88af929-c965-4ef0-8514-7f91b58be9d6'

#converting datetime object to unix time
def unix_time(dt):
    epoch = datetime.utcfromtimestamp(0)
    return ((dt - epoch).total_seconds())

#hitting the API and returning the tide on the given day
def get_tide(lat,lon,dt):
    base_url = 'https://www.worldtides.info/api?heights'
    ut = (unix_time(dt))
    #generating the url
    url = base_url + '&lat=' + str(lat) + '&lon=' + str(lon) + '&start=' + str(ut) + \
    '&length=1000&key=' + os.environ['API_KEY']
    #hitting API
    r = requests.get(url)
    #converting respoiinse to JSON
    tide = json.loads(r.text)
    return tide['heights'][0]['height']

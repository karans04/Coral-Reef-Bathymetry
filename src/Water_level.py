import pandas as pd
import numpy as np


def get_water_level(df):
    """
    Calculate function to represent sea level
    Params - 1. df (Dataframe) - photon depth, lat, lon
    Return np.poly1d - function representing the sea level
    """

    water,lat = [],[]
    #gets just ocean photons
    df = df.loc[df.Conf_ocean == 4]
    if len(df) == 0:
        return None
    #getting photons +- 2 of the median height of photons
    df = df.loc[(df.Height > df.Height.median() - 2) & (df.Height < df.Height.median() + 2)]

    #creating a df with just the latitude and height
    sea_level = pd.DataFrame([df.Height,df.Latitude]).T.dropna()
    sea_level.columns = ['water','latitude']

    #getting photons +- 1.25 of the median height of photons
    sea_level = sea_level.loc[(sea_level.water > sea_level.water.median() -1.25) & (sea_level.water < sea_level.water.median() +1.25)]

    #fitting linear line to remaining points
    z = np.polyfit(sea_level.latitude, sea_level.water,1)
    f = np.poly1d(z)

    #getting points with <2m abs error
    sea_level['abs_diff'] = np.abs(sea_level.water - f(sea_level.latitude))
    sea_level = sea_level.loc[sea_level.abs_diff < 2]
    #fitting a parabolic function to the remaining points
    z2 = np.polyfit(sea_level.latitude, sea_level.water,2)
    f2 = np.poly1d(z2)

    return f2

def adjust_for_speed_of_light_in_water(df, tide_level):
    """
    Adjust photon depth to account for change in the speed of photons
    in air and water.
    Params - 1. df(Dataframe) - photon depth, lat, lon
    Return Dataframe - photon depth, lat, lon
    """
    speed_of_light_air = 300000
    speed_of_light_water = 225000
    coef = speed_of_light_water / speed_of_light_air
    df['Height'] = (df['Height']- tide_level) * coef
    return df

def adjust_for_refractive_index(df):
    """
    Adjust photon depth with regards to tide and refractive index
    Params - 1. df (Dataframe) - photon depth, lat, lon
    	     2. tide_level (int) - tide level on given day
    Return - Dataframe - photon depth, lat, lon
    """
    refractive_index_salt_water = 1.33
    df['Height'] = (df['Height']) / refractive_index_salt_water
    return df

def normalise_sea_level(df):
    """
    Adjusting photons to sea level
    Params - 1. df (Dataframe) - photon depth, lat, lon
    Return - Dataframe - photon depth, lat, lon
             np.poly1d - function representing sea level
    """
    #calculating sea level
    f = get_water_level(df)
    if not f:
        return pd.DataFrame(), None
    #getting ocean and reef photons
    df = df.loc[(df.Conf_ocean == 4) | (df.Conf_land == 4)]
    #adjust photons to sea level
    sea = f(df.Latitude)
    mean_sea = np.mean(sea)
    df.Height = df.Height - mean_sea
    df = df.loc[df.Height < 10]
    return df,f

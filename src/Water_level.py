import pandas as pd
import numpy as np

import src.Water_level as water_level

 #method to predict the water level
def get_water_level(df):
    df = df.loc[df.Conf_ocean == 4]
    #getting photons +- 2 of the median height of photons
    df = df.loc[(df.Height > df.Height.median() - 2) & (df.Height < df.Height.median() + 2)]
    water = []
    lat = []

	#creating a df with just the latitude and height
    sea_level = (pd.DataFrame([df.Height,df.Latitude]).T.dropna())
    sea_level.columns = ['water','latitude']

 	#getting photons +- 1.25 of the median height of photons
    sea_level = sea_level.loc[(sea_level.water > sea_level.water.median() -1.25) & (sea_level.water < sea_level.water.median() +1.25)]


    #fitting linear line to remaining points
    z = np.polyfit(sea_level.latitude, sea_level.water,1)
    f = np.poly1d(z)
    # print(f)
    #getting absolute error for each point
    sea_level['abs_diff'] = np.abs(sea_level.water - f(sea_level.latitude))
    #retaining only points with absolute error less than 2
    sea_level = sea_level.loc[sea_level.abs_diff < 2]
    #fitting a parabolic function to the remaining points
    z2 = np.polyfit(sea_level.latitude, sea_level.water,2)
    f2 = np.poly1d(z2)

    #return the function
    return f2

def adjust_for_speed_of_light_in_water(df):
    speed_of_light_air = 300000
    speed_of_light_water = 225000
    coef = speed_of_light_water / speed_of_light_air
    df['Height'] = df['Height'] * coef
    return df

def adjust_for_refractive_index(df, tide_level):
    refractive_index_salt_water = 1.33
    df['Height'] = (df['Height'] - tide_level) / refractive_index_salt_water
    return df

def normalise_sea_level(df):
    f = water_level.get_water_level(df)
    df = df.loc[(df.Conf_ocean == 4) | (df.Conf_land == 4)]
    sea = f(df.Latitude)
    mean_sea = np.mean(sea)
    df.Height = df.Height - mean_sea
    df = df.loc[df.Height < 10]
    return df,f

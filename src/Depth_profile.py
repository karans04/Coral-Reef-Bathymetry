import os
import pandas as pd
import numpy as np
import math
import geopandas as gpd
import pyproj as proj
from sklearn.cluster import DBSCAN

import src.Water_level as water_level
import src.Coral_Reef as coral_reef
import src.ICESAT_plots as is2_plot
import src.IS2_file as is2File


def create_photon_df(photon_data):
    """
    Creates dataframe from photon data
    Params - 1. photon_data (list) - photon data extracted from h5 file
    Return - DataFrame - containing photon data
    """
    df = pd.DataFrame(photon_data).T
    #sets the column names
    df.columns = ['Height', 'Latitude', 'Longitude','Confidence']
    return df


def individual_confidence(df):
    """
    Gets the confidence of photons for land and ocean
    Params - 1. df (DataFrame) - photon dataframe
    Return - DataFrame - added columns for photon confidence
    """
    # print(df)
    # 0 - land, 1 - ocean, 2 - sea ice, 3 - land ice, 4 - inland water
    df['Conf_land'] = df.apply(lambda x: x.Confidence[0], axis = 1)
    df['Conf_ocean'] = df.apply(lambda x: x.Confidence[1], axis = 1)
    return df

def convert_h5_to_csv(is2_file,laser,out_fp):
    """
    Converts photon data from h5 file to a csv file
    Corrects the photon depths making them relative to sea level, adjusted for refractive index,
    speed of light in water and tide on the day the satellite orbits over reef.

    Params - 1. is2_file (IS2_file) - object representing icesat file
    	     2. laser (str) - laser of satellite we want photons for
	     3. out_fp (str) - path to store csv file
    Return - DataFrame - photon data over reef
    """
    #creates dataframe with photon data
    photon_data = is2_file.get_photon_data(laser)
    df_laser = create_photon_df(photon_data)
    #gets coordinates of bounding box around the coral reef
    coords = is2_file.get_bbox_coordinates()
    min_longitude,min_latitude,max_longitude,max_latitude = coords
    tide_level = is2_file.get_tide()
    #gets the coordinates that are within the coordinates of the bounding box
    df = df_laser.loc[(df_laser.Longitude > min_longitude) & (df_laser.Longitude < max_longitude) &\
     (df_laser.Latitude > min_latitude) & (df_laser.Latitude < max_latitude)]

    #if there are photons in bounding box
    if len(df) != 0:
        #unpacks the confidence array to individual columns
        df = individual_confidence(df)
        #adjusting for sea level, speed of light in water, tide and refractive index
        df,f = water_level.normalise_sea_level(df)
        if len(df) == 0:
            print('No photons')
            return df
        is2_file.set_sea_level_function(f,laser)
        df = water_level.adjust_for_speed_of_light_in_water(df,tide_level)
#         df = water_level.adjust_for_refractive_index(df)
        #writes a dataframe containing just the photon data that is required
        df.to_csv(out_fp)
    else:
        print('No photons')
    return df



def apply_DBSCAN(df, out_path, is2, laser):
    empty_df = pd.DataFrame()
    #threshold below sea level for which we consider reefs
    water_level_thresh = 0.5
    #getting just high confidence land photons
    df = df.loc[(df.Conf_land == 4)]
    if len(df) == 0:
        return empty_df


    # setup your projections
    crs_wgs = proj.Proj(init='epsg:4326') # assuming you're using WGS84 geographic
    crs_bng = proj.Proj(init='epsg:3857') # use a locally appropriate projected CRS

    # then cast your geographic coordinate pair to the projected system
    df['x'], df['y'] = proj.transform(crs_wgs, crs_bng, df.Longitude.values, df.Latitude.values)
    dbscan = DBSCAN(eps = 1, min_samples = 5)

    f = is2.get_sea_level_function(laser)
    sea = df['Latitude'].apply(f)
    mean_sea = np.mean(sea)
    sea -= mean_sea
    df['sea_level'] = sea
    df = df.loc[df.Height < df.sea_level-water_level_thresh]
    x = df[['Height','x']]
    if len(x) == 0:
        return empty_df
    model = dbscan.fit(x)
    labels = model.labels_
    df['labels'] = labels
    reef_ph = df.loc[df.labels >= 0]
    if len(reef_ph) < 1000:
        return empty_df

    df.to_csv(out_path)
    return df




def depth_profile_adaptive(df, out_path,is2,laser):
    """
    Cleans noisy photons from ICESAT output using an adaptive rolling window
    Params - 1. df (DataFrame) - contains all icesat photons
             2. out_path - path to save cleaned photon data
    Return - DataFrame - contains cleaned icesat photons
    """
    #threshold below sea level for which we consider reefs
    water_level_thresh = 0.5

    #generating arrays for the height, latitude
    h,l = [], []

    #iterating through the latitudes 0.0005 degrees at a time
    start = df.Latitude.min()
    end = df.Latitude.max()
    dx = 0.0005

    #getting just high confidence land photons
    df = df.loc[(df.Conf_land == 4)]
    if len(df) == 0:
        return pd.DataFrame()
    df = df.astype({'Latitude': 'float', 'Longitude':'float'})
    #sorting photons by latitude
    df = df.sort_values('Latitude')


    lon_samples = df.sample(min(1000, len(df)))
    z1 = np.polyfit(lon_samples.Latitude, lon_samples.Longitude,1)
    lon_func = np.poly1d(z1)

    f = is2.get_sea_level_function(laser)
    lats = df.Latitude.drop_duplicates()
    sea = f(lats)
    mean_sea = np.mean(sea)



    while start <= end:
        #subsetting photons that fall within window of size dx
#         temp = df.loc[(df.Latitude >= start-((dx/2))) & (df.Latitude <start+((dx/2)))]
        temp = df.loc[(df.Latitude >= start) & (df.Latitude <start+dx)]
        #getting the midpoint latitude of the window
        mean_lat = (temp.Latitude.max() + temp.Latitude.min()) / 2
        #subsetting coral reef photons
        temp = temp.loc[(temp.Height < f(mean_lat) - mean_sea  - water_level_thresh)]
        if len(temp) == 0:
            start += dx
            continue
        #getting the IQR of photons
        uq = temp["Height"].quantile(0.75)
        lq = temp["Height"].quantile(0.25)
        temp = temp.loc[(temp.Height >= lq) & (temp.Height < uq) ]

        #if IQR contains more than 3 photons we proceed with the depth analysis
        if temp.shape[0] > 3:
            #getting depths that we will iterate throguh
            min_depth = math.ceil(temp.Height.min()-1)
            max_depth = min(0,math.ceil(temp.Height.max()))
            median_depth = pd.DataFrame()


            #iterating through intervals of 0.5m at a time
            for x in range(min_depth,max_depth):
                for y in range(2):
                    #subsetting photons within each 0.5m interval
                    depth_intervals = temp.loc[(temp.Height >= x + (y/2)) & (temp.Height < x+ ((y+1)/2))]
                    #if the interval contains one or more photons we will store the photon information for future calculations
                    if depth_intervals.shape[0] >=1:
                        median_depth = pd.concat([depth_intervals,median_depth])

            #if more than 2 photons are saved from the previous step we set the median to be the predicted height else nan
            if median_depth.shape[0] >= 2:
                h.append(median_depth.Height.median())
            else:
                h.append(np.nan)

            #append the mid point of the latitude to represent the latitude for the calculated height
            l.append((start + start+dx)/2)
            #move to the next window
            start += dx

        #if the IQR does not contain more than 3 photons we use an adaptive window
        else:
            #we have already completed the first iteration by checking if the window has more than 3 photons
            i = 2
            #boolean flag to check if we have met the requirements to make a prediction
            bool_check = False
            #saving the starting latitude
            ts = start
            #adaptive window check done at max 4 times
            while i <= 4:
                #subset data in the window
                temp = df.loc[(df.Latitude >= start-(i*(dx/2))) & (df.Latitude <start+(i*(dx/2)))]
                #get the midpoint of the latitudes in the window
                mean_lat = (temp.Latitude.max() + temp.Latitude.min()) / 2
                #get coral reef photons
                temp = temp.loc[(temp.Height < f(mean_lat) - mean_sea  - water_level_thresh)]

                #setting counter to move to the next adaptive window
                i+=1
                #check if there are more than 30 photons in the window
                if temp.shape[0] > 30:

                    #find depths through which we will iterate
                    min_depth = math.ceil(temp.Height.min()-1)
                    max_depth = min(0,math.ceil(temp.Height.max()))
                    median_depth = pd.DataFrame()
                    #iterate through depths of 0.5m at a time
                    for x in range(min_depth,max_depth):
                        for y in range(2):
                            depth_intervals = temp.loc[(temp.Height >= x + (y/2)) & (temp.Height < x+ ((y+1)/2))]
                            #if any depth interval has 2 or more photons, we will store that information for future use
                            if depth_intervals.shape[0] >=1:
                                median_depth = pd.concat([depth_intervals,median_depth])

                    #if we had more than 2 photons saved from the previous step, we will preduct the median of those photons as the height
                    if median_depth.shape[0] >= 2:
                        #set the boolean flag to true
                        bool_check = True
                        h.append(median_depth.Height.median())
                        #latitude for the given depth prediction is represented by the latitude midpoint
                        l.append((start + start+dx)/2)
                        i = 5

            #if we did not meet any of the criteria we will predict nan for the height and the midpoint of the latitude for lat
            if bool_check == False:
                h.append(np.nan)
                l.append((start + start+dx)/2)
            #move to the next window
            start = ts + dx

    #remove noise
    if len(h) >= 2:
        h = remove_depth_noise(h)
        #creating dataframe with the depth and latitudes
        depth = pd.DataFrame([h,l,lon_func(l)]).T
        depth.columns = ['Height','Latitude','Longitude']


        #disregards files with less than 10 depth predictions
        if depth.dropna().shape[0] >= 15:
            depth.to_csv(out_path)
            return depth
    return pd.DataFrame()



def remove_depth_noise(depths):
    """
    Removes noise from cleaned data
    Params - 1. depths (DataFrame) - cleaned photon predictions with some noise
    Return - DataFrame - cleaned from noisy predictions
    """
    prev = depths[0]
    curr = depths[1]
    #only retains photons that have a prediction before and after
    for i in range(1, len(depths) -1):
        next = depths[i+1]
        if np.isnan(prev) and np.isnan(next):
            depths[i] = np.nan
        prev,curr = curr,next
    return depths


def combine_is2_reef(is2, depths):
    """
    Combining ICESAT-2 output with cleaned photons
    Creates data output for plots
    Params - 1. is2 (DataFrame) - ICESAT-2 photons
    	     2. depths (DataFrame) - cleaned photons
    """
    is2 = is2[['Latitude', 'Longitude', 'Height']]
    depths = depths[['Latitude', 'Longitude','Height' ,'labels']]
    #casting to float
    is2 = is2.astype({'Latitude': 'float', 'Longitude':'float'})
    depths = depths.astype({'Latitude': 'float', 'Longitude':'float'})
    #rounding to 4dp
    is2['Latitude'] = np.round(is2['Latitude'], decimals=4)
    is2['Longitude'] = np.round(is2['Longitude'], decimals=4)
    # is2['Photon_depth'] = is2['Height']
    depths['Latitude'] = np.round(depths['Latitude'], decimals=4)
    depths['Longitude'] = np.round(depths['Longitude'], decimals=4)
    # depths['Predicted_depth'] = depths['Height']
    #merging on lat,lon
    merged =  is2.merge(depths, on = ['Latitude', 'Longitude', 'Height'], how = 'outer')
    merged['labels'] = merged['labels'].fillna(-2)
    return merged


def process_h5(reef, is2_file):
    """
    Calculate depth predictions for a single ICESAT-2 file
    Params - 1. reef (Coral_Reef) - reef ICESAT-2 is orbiting over
             2. ise_file (IS2_file) - ICESAT-2 file we are processing
    """
    #gets directories to save outfiles to
    icesat_fp, proc_fp, images_fp,data_plots_path = reef.get_file_drectories()
    #gets reef name and is2 filename without extension
    reef_name = reef.get_reef_name()
    is2_file_tag = is2_file.get_file_tag()
    #looping through each strong laser
    for laser in is2_file.get_strong_lasers():
        print(laser)
    	#path for csv file containing raw photon data
        photon_fn = '{reef_name}_photons_{h5_fn}_{laser}.csv'.format(reef_name=reef_name, h5_fn=is2_file_tag, laser=laser)
        photons_path = os.path.join(icesat_fp, photon_fn)
        #loading raw photon data if it exists, else extracting it from h5 file
        # photons = convert_h5_to_csv(is2_file,laser,photons_path)
        if not os.path.exists(photons_path):
            photons = convert_h5_to_csv(is2_file,laser,photons_path)
        else:
            photons = pd.read_csv(photons_path)
            is2_file.metadata = is2_file.load_json()
        if len(photons) == 0:
            continue
    	#if length of photons is zero, move onto next laser
        print('Number of ICESAT-2 Photons in {laser} is {reef_length}'.format(laser=laser, reef_length=str(len(photons))))
    	#calculates predicted depths and saves file to the following path
        depths_fn =  '{reef_name}_{h5_fn}_{laser}.csv'.format(reef_name=reef_name,h5_fn=is2_file_tag,laser=laser)
        processed_output_path = os.path.join(proc_fp,depths_fn)
        depth = apply_DBSCAN(photons,processed_output_path,is2_file,laser)

        print('Number of reef Photons in {laser} after cleaning is {reef_length}'.format(laser=laser, reef_length=str(len(depth))))
        if len(depth) != 0:
	    #combines ICESAT-2 output with depth predictions
            out_df = combine_is2_reef(photons, depth)
            data_plots_fn =  '{reef_name}_{h5_fn}_{laser}_plots.csv'.format(reef_name=reef_name,h5_fn=is2_file_tag,laser=laser)
            out_df.to_csv(os.path.join(data_plots_path,data_plots_fn))
	    #plot predicted depths with ICESAT photons
            is2_plot.p_is2(out_df,is2_file,laser,images_fp)

def get_depths(reef):
    """
    Wrapper function that takes in the reef and outputs the depth profile of each reef
    Params - 1. reef (Coral_Reef) - reef over which the ICESAT satelitte is orbitting
    """
    h5_dir = os.path.join(reef.get_path(),'H5')
    #looping through each h5 file and generating cleaned photon data
    for h5_fn in os.listdir(h5_dir):
        if h5_fn.endswith('.h5'):
            print(h5_fn)
            is2 = is2File.IS2_file(h5_dir, h5_fn,reef.bbox_coords)
            process_h5(reef, is2)

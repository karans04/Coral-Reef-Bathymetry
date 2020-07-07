#importing required packages
import os
import pandas as pd
import numpy as np
import math
import geopandas as gpd

#importing other files required
import src.Water_level as water_level
import src.Coral_Reef as coral_reef
import src.ICESAT_plots as is2_plot
import src.IS2_file as is2File

#extracts the height, lat,lon and confidence of photons
def create_photon_df(photon_data):
    df = pd.DataFrame(photon_data).T
    #sets the column names
    df.columns = ['Height', 'Latitude', 'Longitude','Confidence']
    return df

#splits the confidence array into individual columns
def individual_confidence(df):
    # 0 - land, 1 - ocean, 2 - sea ice, 3 - land ice, 4 - inland water
    df['Conf_land'] = df.apply(lambda x: x.Confidence[0], axis = 1)
    df['Conf_ocean'] = df.apply(lambda x: x.Confidence[1], axis = 1)
    return df

#method to get only the required parts of the h5 file and outputting it as a csv
def convert_h5_to_csv(is2_file,laser,out_fp):
    photon_data = is2_file.get_photon_data(laser)
    coords = is2_file.get_bbox_coordinates()
    df_laser = create_photon_df(photon_data)
    min_longitude,min_latitude,max_longitude,max_latitude = coords
    tide_level = is2_file.get_tide()
    #gets the coordinates that are within the coordinates of the bounding box
    df = df_laser.loc[(df_laser.Longitude > min_longitude) & (df_laser.Longitude < max_longitude) &\
     (df_laser.Latitude > min_latitude) & (df_laser.Latitude < max_latitude)]

    # print(df)
    #no photons in bounding box
    if len(df) != 0:
        #unpacks the confidence array to individual columns
        df = individual_confidence(df)
        #adjusting for sea level, speed of light in water, tide and refractive index
        df,f = water_level.normalise_sea_level(df)
        is2_file.set_sea_level_function(f,laser)
        df = water_level.adjust_for_speed_of_light_in_water(df)
        df = water_level.adjust_for_refractive_index(df, tide_level)
        #writes a dataframe containing just the photon data that is required
        df.to_csv(out_fp)
    return df


def depth_profile_adaptive(df, out_path):
    #threshold below sea level for which we consider reefs
    water_level_thresh = -0.5

    #generating arrays for the height, latitude
    h = []
    l = []

    #iterating through the latitudes 0.0005degrees at a time
    start = df.Latitude.min()
    end = df.Latitude.max()
    dx = 0.0005

    #getting just high confidence land photons
    df = df.loc[(df.Conf_land == 4)]
    df = df.astype({'Latitude': 'float', 'Longitude':'float'})
    #sorting photons by latitude
    df = df.sort_values('Latitude')

    #getting line for longitudes
    lon_samples = df.sample(min(1000, len(df)))
    z1 = np.polyfit(lon_samples.Latitude, lon_samples.Longitude,1)
    lon_func = np.poly1d(z1)

    while start <= end:
        #subsetting photons that fall within window of size dx
#         temp = df.loc[(df.Latitude >= start-((dx/2))) & (df.Latitude <start+((dx/2)))]
        temp = df.loc[(df.Latitude >= start) & (df.Latitude <start+dx)]
        #getting the midpoint latitude of the window
        mean_lat = (temp.Latitude.max() + temp.Latitude.min()) / 2
        #subsetting photons 1m below sea level
        temp = temp.loc[(temp.Height <  water_level_thresh)]

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
                #get photons more than 1m below sea level
                temp = temp.loc[(temp.Height < water_level_thresh)]

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

                    #if we had more than photons saved from the previous step, we will preduct the median of those photons as the height
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
    h = remove_depth_noise(h)
    #creating dataframe with the depth and latitudes
    depth = pd.DataFrame([h,l,lon_func(l)]).T
    depth.columns = ['Height','Latitude','Longitude']


	#disregards files with less than 10 depth predictions
    if depth.dropna().shape[0] >= 15:
        depth.to_csv(out_path)
        return depth
    else:
        return pd.DataFrame()



def remove_depth_noise(depths):
    prev = depths[0]
    curr = depths[1]

    for i in range(1, len(depths) -1):
        next = depths[i+1]
        if np.isnan(prev) and np.isnan(next):
            depths[i] = np.nan
        prev,curr = curr,next

    return depths


def combine_is2_reef(is2, depths):
    is2 = is2.astype({'Latitude': 'float', 'Longitude':'float'})
    depths = depths.astype({'Latitude': 'float', 'Longitude':'float'})
    is2['Latitude'] = np.round(is2['Latitude'], decimals=4)
    is2['Longitude'] = np.round(is2['Longitude'], decimals=4)
    is2['Photon_depth'] = is2['Height']
    depths['Latitude'] = np.round(depths['Latitude'], decimals=4)
    depths['Longitude'] = np.round(depths['Longitude'], decimals=4)
    depths['Predicted_depth'] = depths['Height']
    return is2[['Latitude', 'Longitude', 'Photon_depth']].merge(depths[['Latitude', 'Longitude', 'Predicted_depth']], on = ['Latitude', 'Longitude'], how = 'outer')


def process_h5(reef, is2_file):
    icesat_fp, proc_fp, images_fp,data_plots_path = reef.get_file_drectories()
    reef_name = reef.get_reef_name()
    is2_file_tag = is2_file.get_file_tag()
    #looping through each strong laser
    for laser in is2_file.get_strong_lasers():
        print(laser)

        photon_fn = '{reef_name}_photons_{h5_fn}_{laser}.csv'.format(reef_name=reef_name, h5_fn=is2_file_tag, laser=laser)
        photons_path = os.path.join(icesat_fp, photon_fn)
        #creates output directory if it does not already exist
        if not os.path.exists(photons_path):
            photons = convert_h5_to_csv(is2_file,laser,photons_path)
        else:
            photons = pd.read_csv(photons_path)
            is2_file.metadata = is2_file.load_json()
        print('Number of ICESAT-2 Photons in {laser} is {reef_length}'\
                .format(laser=laser, reef_length=str(len(photons))))
        if len(photons) == 0:
            continue
        depths_fn =  '{reef_name}_{h5_fn}_{laser}.csv'\
                        .format(reef_name=reef_name,h5_fn=is2_file_tag,laser=laser)
        processed_output_path = os.path.join(proc_fp,depths_fn)

        #calculates the depth profile
        depth = depth_profile_adaptive(photons,processed_output_path)

        print('Number of reef Photons in {laser} after cleaning is {reef_length}'\
                .format(laser=laser, reef_length=str(len(depth))))
        if len(depth) != 0:
            out_df = combine_is2_reef(photons, depth)
            data_plots_fn =  '{reef_name}_{h5_fn}_{laser}_plots.csv'\
                            .format(reef_name=reef_name,h5_fn=is2_file_tag,laser=laser)
            out_df.to_csv(os.path.join(data_plots_path,data_plots_fn))
            is2_plot.p_is2(out_df,is2_file,laser,images_fp)


#wrapper function that takes in the reef and outputs the depth profile of each reef with plots if requested
def get_depths(reef):

    h5_dir = os.path.join(reef.get_path(),'H5')
    #looping through each h5 file
    for h5_fn in os.listdir(h5_dir):
        if h5_fn.endswith('.h5'):
            is2 = is2File.IS2_file(h5_dir, h5_fn,reef.bbox_coords)

            process_h5(reef, is2)
    

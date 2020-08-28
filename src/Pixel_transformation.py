from bs4 import BeautifulSoup
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import rasterio
from rasterio import plot
import numpy as np
import math
from datetime import datetime
import glob
from pathlib import Path
from tqdm import tqdm

#machine learning packages
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

#importing other python files
import src.Tide_API as tide
import src.Depth_profile as depth
import src.Reef_plots as reef_plots
import src.Sentinel_API as sentinel
import src.Sentinel2_image as s2_img


def prep_df(sf,fp,crs):
    """
    Loads in depth prediction data and converts lat,lon to crs of Sentinel-2 images
    Params - 1. sf (Sentinel2_image) - Object for sentinel image
             2. fp (str) - path of depth predictions
             3. crs (dict) - crs of Sentinel-2 image
    Return - DataFrame - depth predictions
    """

    #reading in the depths
    df = pd.DataFrame.from_csv(fp)
    #adjusting the heights based on the tide on the given day
    df.Height = df.Height + sf.get_tide()
    # print(df.columns)
    df = df[['Latitude', 'Longitude', 'Height', 'labels']]

    df = df.loc[df['labels'] > 0]
    #converts the lat and lon to the crs of the image
    df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
    df['Coordinates'] = df.Coordinates.apply(Point)
    df = gpd.GeoDataFrame(df, geometry = 'Coordinates')
    df.crs = {'init': 'epsg:4326'}
    df = df.to_crs(crs)
    # print(df)
    return df

def load_ICESAT_predictions(icesat_proc_path,sf):
    """
    Load in depth predictions over all ICESAT orbits and tracks
    Params - 1. icesat_proc_path (str) - path to load depth predictions from
             2. sf (Sentinel2_image) - object for sentinel image
    """
    #appends depth predictions to training dataframe
    train = pd.DataFrame()
    for fn in os.listdir(icesat_proc_path):
        if fn.endswith('.csv'):
            train_path = os.path.join(icesat_proc_path, fn)
            train = pd.concat([train,prep_df(sf,train_path,sf.get_crs())])
    return train

def get_regressor(reef,sf):
    """
    Trains a regression model
    Params - 1. reef (Coral_Reef) - object representing coral reef
             2. sf (Sentinel2_image) - object representing sentinel image
    Return - lambda function - to predict depths given pixel value
             dict - metadata
             DataFrame - training data (pixel value and depth prediction
    """
    #loads in the different band images required
    imgs = sf.load_sentinel()
    meta = sf.get_meta()

    #creating a training dataset using ICESAT 2 depth profile
    icesat_proc_path = reef.get_processed_output_path()
    train = load_ICESAT_predictions(icesat_proc_path,sf)

    #drop any nan rows
    train = train.dropna()
    image_resolution = 10
    train['x'] = (train.Coordinates.x // image_resolution) * image_resolution
    train['y'] = (train.Coordinates.y // image_resolution) * image_resolution
    train.groupby(['x','y'])['Height'].median()
    #creates the masking threshold for band 8 to mask land and clouds
    b8_pix = imgs[3]
    b8_pix = b8_pix.astype('float')
    b8_pix[b8_pix == 0] =np.NaN
    mask_thresh = np.nanmedian(b8_pix) + (0.75*np.nanstd(b8_pix))
    sf.meta['mask_thresh'] = mask_thresh
    # print(sf.meta['mask_thresh'])
    def get_pixel_val(coord):
        """
        Get pixel value given a set of coordinates
        Params - 1. coord (Point) - point of interest
        Return int - pixel value at point
        """
        x_index = int((coord.x - meta['ulx']) // meta['xdim'])
        y_index = int((coord.y - meta['uly']) // (meta['ydim']))
        img_shape = imgs[0][0].shape
        # print(coord.x,x_index, coord.y, y_index)
        if y_index < img_shape[0] and x_index < img_shape[1]:
            pix_val =  [data[0][y_index][x_index] for data in imgs]
        else:
            pix_val = [0 for data in imgs]
        return pix_val

    def extract_pixel_cols(df):
        """
        Extracts band values for image
        Params - 1. df (DataFrame) - depth predictions of ICESAT-2
        Return - DataFrame - pixel values added for each point
        """
        df['Pixels'] = df.Coordinates.apply(get_pixel_val)
        df['b2'] = df.Pixels.apply(lambda x: (x[0]))
        df['b3'] = df.Pixels.apply(lambda x: (x[1]))
        df['b8'] = df.Pixels.apply(lambda x: (x[3]))
        print(sf.meta['mask_thresh'])
        #converts points with pixel values above the mask threshold to nan
        df['mask'] = df.b8.apply(lambda x: False if x < mask_thresh else True)
        return df
    # display(train)
    train = extract_pixel_cols(train)
    # print(train)
    #remove cloud pixels and zeroed out pixels
    train = train.loc[(train.b3 != 0) & (train.b2 != 0)]

    train = train.loc[train['mask'] == False]
    # print(train.shape)
    #calculates the log difference between band2 and band3 pixels
    delta = 0.0001
    bp = {'B02': 0, 'B03': 0}
    sf.meta['min_pix'] = bp
    train['b2'] = train['b2'].apply(lambda x: max(delta,x - bp['B02']))
    train['b3'] = train['b3'].apply(lambda x: max(delta,x - bp['B02']))
    # print(train)
    train['diff'] = train.apply(lambda x: (math.log(x['b2']) - math.log(x['b3'])), axis = 1)

    train_data_cleaned = remove_log_outliers(train)

    #gets the column that we are trying to predict and performs a train test split
    x = train_data_cleaned.loc[:,['Height']]
    y = train_data_cleaned.loc[:,['diff']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state =3)

    #fit a linear regression ridge model to our data
    reg = linear_model.HuberRegressor()
#     reg = LinearRegression()
    reg.fit(X_train,y_train)

    #predicts values for our test set
    preds = reg.predict(X_test)
    mse_train = mean_squared_error(y_train, reg.predict(X_train))
    mse_test = mean_squared_error(y_test,preds)

    # meta['mse_train'] = mse_train
    # meta['mse_test'] = mse_test
    print('mse train ', str(mse_train))
    print('mse test ', str(mse_test))

    #switches independent and dependent variable and generates function to predict depth
    m = reg.coef_[0]
    c = reg.intercept_
    line = lambda x: (x-c) /m

    out = train_data_cleaned[['x','y','b2','b3','diff','Height']]
    # meta['correlation'] = pearsonr(X_train.values, y_train.values)
    return line, meta, out


def remove_log_outliers(data):
    """
    Removes outliers for training data
    Params - 1. data (DataFrame) - training data (pixels and depth predictions
    Return - DataFrame - removed outlieres at 0.5m intervals
    """
    new_data = data.reset_index()
    curr = 0
    #get training data at 0.5m intervals and removes outliers
    intervals = 0.5
    while curr > min(data['Height']):
        temp = new_data.loc[(new_data['Height'] < curr) & (new_data['Height'] > curr - intervals)]
        if len(temp) >= 1:
            q1, q3= np.percentile(temp['diff'],[25,75])
            iqr = q3-q1
            outlier_range = 1.5 * iqr
            lower_outlier = q1 - outlier_range
            upper_outlier = q3 + outlier_range
            drop_indices = temp.loc[(temp['diff'] < lower_outlier) | (temp['diff'] > upper_outlier)].index
            new_data = new_data.drop(drop_indices)

        curr -= intervals
    return new_data

def predict_reef(reg, sf,master_df):
    """
    Predict the depth of the reef using colour pixel values
    Params - 1. reg (lambda) - predict depth using pixel values
             2. sf (Sentinel2_image) - Object representing sentinel image
             3. master_df (DataFrame) - contains depth predictions from all sentinel images
    Return - str - path of out file
           - DataFrame - containing depth predictions from all sentinel images
    """
    #creating lists to store required values
    x,y,height, pix,log_pix = [],[],[],[],[]
    meta = sf.get_meta()
    #loads in the required images
    b2_pix,b3_pix,_,b8_pix = meta['imgs']
    print(b2_pix.shape, b3_pix.shape, b8_pix.shape)
    #stores the masking threshold
    mask_thresh = meta['mask_thresh']
    tide_level = sf.get_tide()
    bbox_coords = sf.read_gjson()['geometry'][0].bounds
    #looping through the image coordinates
    for j in tqdm(range(len(b2_pix[0]))):
        for i in range(len(b2_pix[0][0])):
            b2,b3,b8 = b2_pix[0][j][i], b3_pix[0][j][i],b8_pix[0][j][i]
            if b2 and b3:
                #generating x,y coordinate for pixel
                x_coord = bbox_coords[0] + ((i)* meta['xdim']) + 5
                y_coord = bbox_coords[3] + ((j)* meta['ydim']) - 5
                p = Point((x_coord, y_coord))
                #getting the normalised band values
                bp = meta['min_pix']
                delta = 0.0001
                band_2 = max(delta,b2 - bp['B02'])
                band_3 = max(delta,b3 - bp['B03'])
                band_8 = b8
                #storing the normalised pixel value
                pix.append([band_2,band_3,band_8])
                #if the band 8 value is higher than the threshold we predict nan for the height else the depth adjust with the tide
                if band_8 > mask_thresh :
                    height.append(np.nan)
                else:
                    x_feat = math.log(band_2) -  math.log(band_3)
                    pred = (reg(x_feat) - tide_level)
                    if pred >= -30 and pred <= 10:
                        height.append(pred)
                    else:
                        height.append(np.nan)

                x.append(x_coord)
                y.append(y_coord)

    #creating a dataframe with the output information and save that df as a csv
    df = pd.DataFrame([x,y,height,pix,log_pix]).T
    df.columns = ['x','y','Height','normalised_pixel', 'diff']
    #adds predictions to master_df
    if 'x' not in master_df.columns:
        master_df['x'] = df.x
    if 'y' not in master_df.columns:
        master_df['y'] = df.y
    dt = sf.get_date()
    master_df[str(dt.strftime("%Y/%m/%d"))] = df.Height

    out_fn = '{reef_name}_out_{dt}.csv'.format(reef_name = sf.reef_name, dt = dt.strftime("%Y%m%d%H%M%S"))
    out_fp = os.path.join(sf.depth_preds_path, out_fn)
    df.to_csv(out_fp)
    return out_fp,master_df



def all_safe_files(reef):
    """
    Depth predictions for all sentinel images over a coral reef
    Params - 1. reef (Coral_Reef) - Object representing the reef of interest
    Return - dict - contains regressor, metadata, training data for each sentinel image
    """
    datum = {}
    medians, variances = [], []
    master_df = pd.DataFrame()

    median_threshold, variance_threshold = 1, 1
    #gets path of safe files
    reef_path = reef.get_path()
    reef_name = reef.get_reef_name()
    safe_files = os.path.join(reef_path, 'SAFE files')
    #gets bounding box of reef
    coords = reef.get_bounding_box()
    #iterating through all safe files
    for sf in os.listdir(safe_files):
        if sf.endswith('.SAFE'):
            #creates object to represent sentinel image
            sf_path = os.path.join(reef_path, sf)
            safe_file = s2_img.Sentinel2_image(sf_path,coords)
            #get out file directories
            imgs_path, depth_preds_path, training_data_path = safe_file.get_file_directories()
            #fits the regressor with training data
            r,m,d = get_regressor(reef,safe_file)

            #checks if median and variances is between required threshold to be considered a valid image
            sample_median = np.median(d['diff'])
            sample_variance = np.var(d['diff'])
            medians.append(sample_median)
            variances.append(sample_variance)
            if (sample_median < median_threshold and sample_median > -median_threshold) and \
            sample_variance < variance_threshold:
                datum[sf] = (r,m,d)
                #predict depth of the coral reef
                training_data_out_fn = '{reef_name}_training_data{date}.csv'.\
                                    format(reef_name = reef_name, date = safe_file.get_date().strftime("%Y%m%d%H%M%S"))
                d.to_csv(os.path.join(training_data_path, training_data_out_fn))
                preds,master_df = predict_reef(r, safe_file,master_df)
                #plot predictions for each sentinel image
                reef_plots.plot_reefs(preds,d,safe_file,r)
    preds = master_df.drop(['x','y'], axis = 1)
    len_preds = len(preds.columns)
    master_df['median'] = preds.apply(lambda x: np.median(x.dropna()) if len(x.dropna()) > int(0.75*len_preds) else np.nan, axis = 1)
    master_df['mean'] = preds.apply(lambda x: np.mean(x.dropna()) if len(x.dropna()) > int(0.75*len_preds) else np.nan, axis = 1)
    preds['max'] = preds.apply(max, axis = 1)
    preds['min'] = preds.apply(min, axis = 1)
    master_df['range'] = preds['max'] - preds['min']
    median_df = master_df[['x','y','median']]
    reef_plots.aggregate_plot(d,median_df,safe_file,'median')
    #save file with data of all sentinel images
    master_df.to_csv('{outpath}/{reef_name}.csv'.format(outpath = reef.get_outpath(), reef_name=reef_name))
    #plots about training data
    reef_plots.plot_median_variance_graph(medians, variances, median_threshold, variance_threshold,imgs_path)
    reef_plots.corr_plot(datum,reef_name,imgs_path)
    return datum

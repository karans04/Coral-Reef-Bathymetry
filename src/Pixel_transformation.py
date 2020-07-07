#importing required packages
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
    #reading in the depths
    df = pd.DataFrame.from_csv(fp)
    #adjusting the heights based on the tide on the given day
    df.Height = df.Height + sf.get_tide()

    #converts the lat and lon to the crs of the image
    df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
    df['Coordinates'] = df.Coordinates.apply(Point)
    df = gpd.GeoDataFrame(df, geometry = 'Coordinates')
    df.crs = {'init': 'epsg:4326'}
    df = df.to_crs(crs)
    return df

def load_ICESAT_predictions(icesat_proc_path,sf):
    train = pd.DataFrame()
    for fn in os.listdir(icesat_proc_path):
        train_path = os.path.join(icesat_proc_path, fn)
        train = pd.concat([train,prep_df(sf,train_path,sf.get_crs())])
    return train


#method that trains a regression model and returns the same
def get_regressor(reef,sf):
    #loads in the different band images required
    imgs = sf.load_sentinel()
    meta = sf.get_meta()

    #creating a training dataset using ICESAT 2 depth profile

    icesat_proc_path = reef.get_processed_output_path()
    train = load_ICESAT_predictions(icesat_proc_path,sf)


    #drop any nan rows
    train = train.dropna()
    train['x'] = train.Coordinates.x
    train['y'] = train.Coordinates.y
    #creates the masking threshold for band 8 to mask land and clouds
    b8_pix = imgs[3]
    mask_thresh = np.median(b8_pix) + (np.std(b8_pix))
    sf.meta['mask_thresh'] = mask_thresh
    #method to get just the pixel values of our bounding box
    def get_pixel_val(coord):
        x_index = int((coord.x - meta['ulx']) // meta['xdim'])
        y_index = int((coord.y - meta['uly']) // (meta['ydim']))
        return [data[0][y_index][x_index] for data in imgs]

    #extracts the different band values for our image
    def extract_pixel_cols(df):
        df['Pixels'] = df.Coordinates.apply(get_pixel_val)
        df['b2'] = df.Pixels.apply(lambda x: (x[0]))
        df['b3'] = df.Pixels.apply(lambda x: (x[1]))
        df['b8'] = df.Pixels.apply(lambda x: (x[3]))
        #converts points with pixel values above the mask to nan
        df['mask'] = df.b8.apply(lambda x: False if x < mask_thresh else True)
        return df

    train = extract_pixel_cols(train)
    #remove cloud pixels and zerod out pixels
    train = train.loc[(train.b3 != 0) & (train.b2 != 0)]
    train = train.loc[train['mask'] == False]

    delta = 0.0001
    bp = {'B02': min(train.b2), 'B03': min(train.b3)}
    sf.meta['min_pix'] = bp
    train['b2'] = train['b2'].apply(lambda x: max(delta,x - bp['B02']))
    train['b3'] = train['b3'].apply(lambda x: max(delta,x - bp['B02']))
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

    m = reg.coef_[0]
    c = reg.intercept_
    line = lambda x: (x-c) /m

    out = train_data_cleaned[['x','y','b2','b3','diff','Height']]
    # meta['correlation'] = pearsonr(X_train.values, y_train.values)
    return line, meta, out


def remove_log_outliers(data):
    new_data = data.reset_index()
    curr = 0
    while curr > min(data['Height']):
        temp = new_data.loc[(new_data['Height'] < curr) & (new_data['Height'] > curr - 0.5)]
        if len(temp) >= 1:
            q1, q3= np.percentile(temp['diff'],[25,75])
            iqr = q3-q1
            outlier_range = 1.5 * iqr
            lower_outlier = q1 - outlier_range
            upper_outlier = q3 + outlier_range
            drop_indices = temp.loc[(temp['diff'] < lower_outlier) | (temp['diff'] > upper_outlier)].index

            new_data = new_data.drop(drop_indices)

        curr -= 1
    return new_data

#method to predict the depth of the rest of the reef
def predict_reef(reg, sf,master_df):
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

                x_coord = bbox_coords[0] + ((i)* meta['xdim']) + 5
                y_coord = bbox_coords[3] + ((j)* meta['ydim']) - 5
                p = Point((x_coord, y_coord))
            #checking if the point is within the reef
    #         if sf.read_gjson().loc[0,'geometry'].contains(p):
    #         # print(b2,b3)
    #         # if b2 != 0  and b3 != 0:
                #getting the band values minus the dark pixel value
                bp = meta['min_pix']
                delta = 0.0001
                band_2 = max(delta,b2 - bp['B02'])
                band_3 = max(delta,b3 - bp['B03'])
                band_8 = b8
                #storing the normalised pixel value
                pix.append([band_2,band_3,band_8])
                #if the band 8 value is higher than the threshold we predict nan for the height
                if band_8 > mask_thresh:
                    height.append(np.nan)
                    x.append(x_coord)
                    y.append(y_coord)
                    continue
                #else we pass the band values into the regressor and adjust with the tide on the day
                x.append(x_coord)
                y.append(y_coord)
                x_feat = math.log(band_2) -  math.log(band_3)

                pred = (reg(x_feat) - tide_level)
                height.append(pred)
    #creating a dataframe with the output information and save that df as a csv
    df = pd.DataFrame([x,y,height,pix,log_pix]).T
    df.columns = ['x','y','Height','normalised_pixel', 'diff']
    if 'x' not in master_df.columns:
        master_df['x'] = df.x
    if 'y' not in master_df.columns:
        master_df['y'] = df.y
    dt = sf.get_date().strftime("%Y%m%d%H%M%S")
    master_df[sf.get_safe_file()] = df.Height

    out_fn = '{reef_name}_out_{dt}.csv'.format(reef_name = sf.reef_name, dt = dt)
    out_fp = os.path.join(sf.depth_preds_path, out_fn)
    df.to_csv(out_fp)
    return out_fp,master_df



def all_safe_files(reef):
    datum = {}
    medians = []
    variances = []
    median_threshold = 1
    variance_threshold = 1

    reef_path = reef.get_path()
    safe_files = os.path.join(reef_path, 'SAFE files')
    coords = reef.get_bounding_box()
    master_df = pd.DataFrame()
    reef_name = reef.get_reef_name()
    for sf in os.listdir(safe_files):
        if sf.endswith('.SAFE'):
            sf_path = os.path.join(reef_path, sf)
            safe_file = s2_img.Sentinel2_image(sf_path,coords)
            imgs_path, depth_preds_path, training_data_path = safe_file.get_file_directories()
            r,m,d = get_regressor(reef,safe_file)

            sample_median = np.median(d['diff'])
            sample_variance = np.var(d['diff'])
            medians.append(sample_median)
            variances.append(sample_variance)

            if (sample_median < median_threshold and sample_median > -median_threshold) and \
            sample_variance < variance_threshold:
                datum[sf] = (r,m,d)
                training_data_out_fn = '{reef_name}_training_data{date}.csv'.\
                                    format(reef_name = reef_name, date = safe_file.get_date().strftime("%Y%m%d%H%M%S"))
                d.to_csv(os.path.join(training_data_path, training_data_out_fn))
                preds,master_df = predict_reef(r, safe_file,master_df)
                reef_plots.plot_reefs(preds,d,safe_file,r)
    master_df.to_csv('{outpath}/{reef_name}.csv'.format(outpath = reef.get_outpath(), reef_name=reef_name))
    reef_plots.plot_median_variance_graph(medians, variances, median_threshold, variance_threshold,imgs_path)
    reef_plots.corr_plot(datum,reef_name,imgs_path)
    return datum

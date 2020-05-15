#importing required packages
from bs4 import BeautifulSoup
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from rasterio import plot
import math
from datetime import datetime
import glob
from pathlib import Path
from tqdm import tqdm

#machine learning packages
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

#plotting packages
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import patches


#importing other python files
import src.Tide_API as tide
import src.Depth_profile as depth

#method to get the metadata of the images
def get_metadata(fp,meta):
    #opening file containing metadata
    file = open(fp,"r")
    contents = file.read()
    soup = BeautifulSoup(contents,'xml')
    #getting the time of the image and creating a datetime object
    dt = (soup.find('SENSING_TIME').text.split('.')[0].replace('T',''))
    meta['dt'] = datetime.strptime(dt, '%Y-%m-%d%H:%M:%S')

    #getting the latitude and longitude to calculate the tide at the given point
    lat = (meta['coords'][1] + meta['coords'][1])/2
    lon = (meta['coords'][2] + meta['coords'][0])/2
    #getting the tide on the day of the image
    meta['tide_level'] = tide.get_tide(lat,lon,meta['dt'])
    print(meta['tide_level'])

    #getting the crs of the image
    geo_info = soup.find('n1:Geometric_Info')
    meta['crs'] = geo_info.find('HORIZONTAL_CS_CODE').text.lower()
    meta['bbox'] = get_bbox(meta['coords'],meta['crs'])

    #getting the number of rows and columns in the image
    rc = geo_info.find('Size' , {'resolution':"10"})
    meta['rows'] = int(rc.find('NROWS').text)
    meta['cols'] = int(rc.find('NCOLS').text)

    #getting the upper left x and y coordinates
    geo_pos = geo_info.find('Geoposition' , {'resolution':"10"})
    meta['ulx'] = int(geo_pos.find('ULX').text)
    meta['uly'] = int(geo_pos.find('ULY').text)

    #getting the step of the image in the x and y dircetions
    meta['xdim'] = int(geo_pos.find('XDIM').text)
    meta['ydim'] = int(geo_pos.find('YDIM').text)

    file.close()


    #getting a polygon representing the shape of the reef
    meta['polygon'] = read_gjson(meta)
    return meta


def get_bbox(coords,crs):
    #creates a bounding box for just the reef we are interested in
    ul = (coords[0], coords[3])
    br = (coords[2], coords[1])
    bbox = pd.DataFrame([ul,br], columns = ['Longitude','Latitude'])
    bbox['Coordinates'] = list(zip(bbox.Longitude, bbox.Latitude))
    bbox['Coordinates'] = bbox.Coordinates.apply(Point)
    #converts the coordinates of the bounding box to the crs of the image
    bbox = gpd.GeoDataFrame(bbox, geometry = 'Coordinates')
    bbox.crs = {'init': 'epsg:4326'}
    bbox = bbox.to_crs(crs)
    return bbox

#function to read in the polygon representing the shape of the reef
def read_gjson(meta):
    #creating the filepath for the geojson file
    fp = os.path.join(meta['data_dir'], meta['reef_name'], str(meta['reef_name']) +'.geojson')
    #loading in the geojson file into a geopandas dataframe
    df = gpd.read_file(fp)
    #setting the current crs of the dataframe
    df.crs = {'init': 'epsg:4326'}
    #changing the crs to that of the sentinel image
    df = df.to_crs(meta['crs'])
    return df

def prep_df(fp,meta):
    #reading in the depths
    df = pd.DataFrame.from_csv(fp)
    #adjusting the heights based on the tide on the given day
    df.Height = df.Height + meta['tide_level']

    #converts the lat and lon to the crs of the image
    df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
    df['Coordinates'] = df.Coordinates.apply(Point)
    df = gpd.GeoDataFrame(df, geometry = 'Coordinates')
    df.crs = {'init': 'epsg:4326'}
    df = df.to_crs(meta['crs'])
    return df

#method to load in all the images
def get_images(base_fp,meta):
    #select the bands that we want
    bands = ['B02','B03','B04','B08']
    imgs = []
    black = {}
    #loops through the bands
    for b in bands:
        img_path = list(Path(base_fp).glob('**/' + 'IMG_DATA' + '/*'+b+'*.jp2'))[0]

        #reads in image
        band = rasterio.open(img_path, driver = 'JP2OpenJPEG')
        #reads in pixel values
        img = band.read(1)
        #adds delta to each pixel value, so that we never take log(0)
        delta = 0.0001
        img = np.add(img,delta)

        #get the coordinates of the bounding box
        start_x = meta['bbox'].loc[0,'Coordinates'].x
        start_y = meta['bbox'].loc[0,'Coordinates'].y

        sx = int((start_x - meta['ulx']) // meta['xdim'])
        sy = int((start_y - meta['uly']) // meta['ydim'])

        end_x = meta['bbox'].loc[1,'Coordinates'].x
        end_y = meta['bbox'].loc[1,'Coordinates'].y

        ex = int((end_x - meta['ulx']) // meta['xdim'])
        ey = int((end_y - meta['uly']) // meta['ydim'])

        #clip the image to the coordinates of the bounding box
        img = [row[sx:ex] for row in (img[sy:ey])]

        imgs.append(img)
    #return the images
    return imgs

#method that trains a regression model and returns the same
def get_regressor(base_fp,meta):
    #gets the metadata
    fp = os.path.join(base_fp,'MTD_TL.xml')
    meta = get_metadata(fp,meta)

    #creating a training dataset using ICESAT 2 depth profile
    reef = pd.DataFrame()
    icesat_proc_fp = os.path.join(meta['data_dir'], meta['reef_name'], 'Output', 'Data Cleaning','processed-output')
    for fn in os.listdir(icesat_proc_fp):
        training_fp = os.path.join(icesat_proc_fp, fn)
        reef = pd.concat([reef,prep_df(training_fp,meta)])

    #drop any nan rows
    reef = reef.dropna()

    #loads in the different band images required
    imgs = get_images(base_fp,meta)
    meta['imgs'] = imgs
    #creates the masking threshold for band 8 to mask land and clouds
    b8_pix = (meta['imgs'][3])
    meta['mask_thresh'] = np.median(b8_pix) + (np.std(b8_pix))

    #method to get just the pixel values of our bounding box
    def get_pixel_val(coord):
        x_index = int((coord.x - meta['bbox'].Coordinates.x[0]) // meta['xdim'])
        y_index = int((coord.y - meta['bbox'].Coordinates.y[0]) // meta['ydim'])
        return [data[y_index][x_index] for data in imgs]

    #extracts the different band values for our image
    def extract_pixel_cols(df):
        df['Pixels'] = df.Coordinates.apply(get_pixel_val)
        df['b2'] = df.Pixels.apply(lambda x: (x[0]))
        df['b3'] = df.Pixels.apply(lambda x: (x[1]))
        df['b8'] = df.Pixels.apply(lambda x: (x[3]))
        #converts points with pixel values above the mask to nan
        df['mask'] = df.b8.apply(lambda x: x if x < meta['mask_thresh'] else np.nan)
        return df

    reef = extract_pixel_cols(reef)
    #drop all rows that are land or clouds
    reef = reef.dropna(subset = ['mask'])

    #stroes the x and y coordinates in individual columns
    reef['x'] = reef.Coordinates.x
    reef['y'] = reef.Coordinates.y
#     removing training data that are above -2m
#     reef = reef.loc[reef.Height < -2]


    #gets just the required input columns
    y = reef.loc[:,['b2','b3']]

    meta['min_pix'] = {'B02': min(y.b2), 'B03': min(y.b3)}
    bp = meta['min_pix']
    y['b2'] = y['b2'].apply(lambda x: max(1,x - bp['B02']))
    y['b3'] = y['b3'].apply(lambda x: max(1,x - bp['B03']))
    y['diff'] = y.apply(lambda x: (math.log(x['b2']) - math.log(x['b3'])), axis = 1)
    y['Height'] = reef.loc[:,['Height']]
    y['x'] = reef.Coordinates.x
    y['y'] = reef.Coordinates.y

    reef = remove_log_outliers(y)
    #gets the column that we are trying to predict
    y = reef.loc[:,['diff']]
    x = reef.loc[:,['Height']]
    #performs a train test split, putting 33% of the data in the test set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state =3)

    #fit a linear regression ridge model to our data
    from sklearn import linear_model
    reg = linear_model.HuberRegressor()
#     reg = LinearRegression()
    reg.fit(X_train,y_train)

    #predicts values for our test set
    preds = reg.predict(X_test)
    mse_train = mean_squared_error(y_train, reg.predict(X_train))
    mse_test = mean_squared_error(y_test,preds)

    meta['mse_train'] = mse_train
    meta['mse_test'] = mse_test
    print('mse train ', str(meta['mse_train']))
    print('mse test ', str(meta['mse_test']))
    m = reg.coef_[0]

    c = reg.intercept_
    line = lambda x: (x-c) /m


    #get the different band values and height for all x and y coordinates
    out = x.copy()
    out['diff'] = y
    out['x'] = reef['x']
    out['y'] = reef['y']

    meta['correlation'] = pearsonr(X_train.values, y_train.values)
    return line, meta, out


def remove_log_outliers(data):
    new_data = data.reset_index()
    curr = -2
    s = 0
    while curr > min(data['Height']):
        temp = new_data.loc[(new_data['Height'] < curr) & (new_data['Height'] > curr - 0.5)]

        if len(temp) >= 1:
            q1, q3= np.percentile(temp['diff'],[25,75])
            iqr = q3-q1

            outlier_range = 1.5 * iqr
            lower_outlier = q1 - outlier_range
            upper_outlier = q3 + outlier_range

            drop_indices = temp.loc[(temp['diff'] < lower_outlier) | (temp['diff'] > upper_outlier)].index
            s += len(drop_indices)

            new_data = new_data.drop(drop_indices)

        curr -= 1
    return new_data

#method to predict the depth of the rest of the reef
def predict_reef(reg, meta):
    #creating lists to store required values
    x = []
    y = []
    h = []

    pix = []
    log_pix = []

    #loads in the required images
    b2_pix = meta['imgs'][0]
    b3_pix = meta['imgs'][1]
    b8_pix = meta['imgs'][3]

    #stores the masking threshold
    mask_thresh = meta['mask_thresh']

    #looping through the image coordinates
    for j in tqdm(range(len(b2_pix))):
        for i in range(len(b2_pix[0])):
            #getting the x and y coordinates
            x_coord = meta['bbox'].Coordinates.x[0] + ((i)* meta['xdim']) + 5
            y_coord = meta['bbox'].Coordinates.y[0] + ((j)* meta['ydim']) - 5
            p = Point((x_coord, y_coord))
            #checking if the point is within the reef
            if meta['polygon'].loc[0,'geometry'].contains(p):
                #getting the band values minus the dark pixel value
#                 bp = meta['black_val']
                bp = meta['min_pix']
                band_2 = max(1,b2_pix[j][i] - bp['B02'])
                band_3 = max(1,b3_pix[j][i] - bp['B03'])
                band_8 = (b8_pix[j][i])

                #storing the pixel value
                pix.append([b2_pix[j][i], b3_pix[j][i],b8_pix[j][i]])
                #storing the normalised pixel value
                log_pix.append([band_2,band_3,band_8])
                #if the band 8 value is higher than the threshold we predict nan for the height
                if band_8 > mask_thresh:
                    h.append(np.nan)
                    x.append(x_coord)
                    y.append(y_coord)
                    continue
                #else we pass the band values into the regressor and adjust with the tide on the day
                x.append(x_coord)
                y.append(y_coord)
                x_feat = math.log(band_2) -  math.log(band_3)

                pred = (reg(x_feat) - meta['tide_level'])
#                 if pred >= 5:
#                     h.append(np.nan)
#                 else:
                h.append(pred)
    #creating a dataframe with the output information and save that df as a csv
    df = pd.DataFrame([x,y,h,pix,log_pix]).T
    df.columns = ['x','y','height','pixels','normalised_pixel']

    out_fn = meta['reef_name']+'_out_'+ meta['dt'].strftime("%Y%m%d%H%M%S") + '.csv'
    out_fp = os.path.join(meta['outpath'], out_fn)
    df.to_csv(out_fp)
    return out_fp


def all_safe_files(data_dir,reef_name):
    medians = []
    variances = []
    median_threshold = 1
    variance_threshold = 0.5
    datum = {}
    reef_path = os.path.join(data_dir, reef_name)

    predictions_fp = os.path.join(reef_path,'Output', 'Depth Predictions')
    if not os.path.exists(predictions_fp):
        os.mkdir(predictions_fp)

    imgs_fp = os.path.join(predictions_fp, 'Imgs')
    if not os.path.exists(imgs_fp):
        os.mkdir(imgs_fp)

    csv_fp = os.path.join(predictions_fp, 'CSV_files')
    if not os.path.exists(csv_fp):
        os.mkdir(csv_fp)



    safe_files = os.path.join(reef_path, 'SAFE_files')
    coords = depth.get_coords(reef_path)
    for sf in os.listdir(safe_files):
        if not sf.startswith('.'):
            meta = {}
            meta['outpath'] = csv_fp
            meta['img_path'] = imgs_fp
            meta['reef_name'] = reef_name
            meta['coords'] = coords
            meta['data_dir'] = data_dir

            pathlist = Path(safe_files).glob('**/' + sf + '/**/*.jp2')
            img_path = os.path.dirname(list(pathlist)[0])
            granule_path = os.path.dirname(img_path)
            r,m,d = get_regressor(granule_path, meta)
            sample_median = np.median(d['diff'])
            sample_variance = np.var(d['diff'])

            medians.append(sample_median)
            variances.append(sample_variance)

            if (sample_median < median_threshold and sample_median > -median_threshold) and \
            sample_variance < variance_threshold:
                datum[sf] = (r,m,d)
                preds = (predict_reef(r, m))
                plot_reefs(preds,d,m,r)

    plot_median_variance_graph(medians, variances, median_threshold, variance_threshold,meta['img_path'])
    corr_plot(datum,reef_name,meta['img_path'])
    return d

#method to plot the reef depth histogram and a scatter plot of the same
def plot_reefs(fp,data,meta,line):
    df = pd.read_csv(fp)
    #plot histogram of depths
    fig, ax = plt.subplots(1,3,figsize = (28,12))
    df['height'].plot.hist(bins = np.arange(-30,20,1), ax = ax[0])
    ax[0].set_xlabel('Height (m)')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title(meta['reef_name'] + ' Depth Histogram')

    #getting just depths between +- 45m
    df = df.loc[(df.height <= 10) & (df.height >= -25)]
    #creating a color scale at 5m intervals

    cmap = cm.colors.ListedColormap(['black','navy','mediumblue' ,'blue','royalblue', 'dodgerblue',
                                     'skyblue','limegreen',  'lime' , 'yellow'
                                      ,'orange','tomato',
                                     'red','firebrick' ,'maroon'])
    bounds = np.arange(-25,11,2.5)

    norm = BoundaryNorm(bounds,cmap.N)

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
        spacing='proportional', ticks=bounds, boundaries=bounds, format='%.1f')


    #scatter plot of the predicted depths
    pts = ax[2].scatter(x = df.x, y = df.y, c = df.height, s= 1, cmap = cmap, norm = norm)
    #scatter plot of the track lines from the ICESAT 2 data
    ax[2].scatter(x = data.x, y= data.y, s = 3, c = 'black', label = 'ICESAT-2 tracks')
    custom_lines = [Line2D([0], [0], color='black', lw=4)]
    ax[2].legend(custom_lines, ['ICESAT-2 tracks'])
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    ax[2].set_title(meta['reef_name'] + ' Depth Predictions (m)')


    r = 3
    sns.scatterplot(x = data['diff'], y = data.Height, color = 'blue', ax = ax[1])
    sns.lineplot(x = [-r,r], y = [line(-r),line(r)], color = 'black', ax = ax[1])
    ax[1].set_xlabel('Log(Blue Band) - Log(Green Band)')
    ax[1].set_ylabel('Depth')
    xt = (list(ax[1].get_xticks()))[:-1]

    for i,x in enumerate(xt):
        xt[i] = np.round(x,2)
    ax[1].set_title(str(meta['dt'].date()) + ' -> tide - ' + str(meta['tide_level']) + 'm')
    xlim = (-r, r)
    ylim = ( -25,0)
    plt.setp(ax[1], xlim=xlim, ylim=ylim)


    fn = meta['reef_name'] + '-' + str(meta['dt'].date())+ '.png'
    out = os.path.join(meta['img_path'], fn)
    plt.savefig(out)
    plt.close(fig)
    return


def corr_plot(datum,reef_name,outpath):
    r = 6
    num_blocks = int(np.ceil(np.sqrt(len(datum))))
    fig, ax = plt.subplots(num_blocks,num_blocks, figsize = (20,24))
    xlim = (-r, r)
    ylim = ( -25,0)
    plt.setp(ax, xlim=xlim, ylim=ylim)

    axlist = []
    for axl in ax:
        for axl2 in axl:
            axlist.append(axl2)


    day_keys = list(datum.keys())
    for i,dict_item in enumerate(datum.items()):
        d = dict_item[1][2]
        line = dict_item[1][0]
        meta = dict_item[1][1]

        sns.scatterplot(x = d['diff'], y = d.Height, color = 'blue', ax = axlist[i])
        sns.lineplot(x = [-r,r], y = [line(-r),line(r)], color = 'black', ax = axlist[i])
        axlist[i].set_xlabel('Log(Blue Band) - Log(Green Band)')
        axlist[i].set_ylabel('Depth')
        axlist[i].set_title(str(meta['dt'].date()))
        xt = (list(axlist[i].get_xticks()))[:-1]
        for i,x in enumerate(xt):
            xt[i] = np.round(x,2)

    fn = os.path.join(outpath,'corr_plot.png')
    plt.savefig(fn)


def plot_median_variance_graph(medians, variances, median_threshold, variance_threshold,outpath):
    ax=sns.scatterplot(x = medians, y = variances, legend = 'full')
    plt.xlabel('Median pixel value for log difference')
    plt.ylabel('Variance pixel value for log difference')

    ax.add_patch(
        patches.Rectangle(
            xy=(-median_threshold, -variance_threshold),  # point of origin.
            width=2*median_threshold,
            height=2*variance_threshold,
            linewidth=1,
            color='red',
            fill=False
        )
    )

    fn = os.path.join(outpath,'median_vs_variance.png')
    plt.savefig(fn)

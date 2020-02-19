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

#machine learning packages
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score

#plotting packages
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

#method to get the metadata of the images
def get_metadata(fp):
    #opening file containing metadata
    file = open(fp,"r")
    contents = file.read()
    soup = BeautifulSoup(contents,'xml')
    #getting the time of the image and creating a datetime object
    dt = (soup.find('SENSING_TIME').text.split('.')[0].replace('T',''))
    dt = datetime.strptime(dt, '%Y-%m-%d%H:%M:%S')
    #getting the coordinates of the reef from depth_profile.py
    _,coords = depth.get_fn_coords(fp.split('/')[1])

    #getting the crs of the image
    geo_info = soup.find('n1:Geometric_Info')
    crs = geo_info.find('HORIZONTAL_CS_CODE').text.lower()

    #getting the number of rows and columns in the image
    rc = geo_info.find('Size' , {'resolution':"10"})
    rows = int(rc.find('NROWS').text)
    cols = int(rc.find('NCOLS').text)

    #getting the upper left x and y coordinates
    geo_pos = geo_info.find('Geoposition' , {'resolution':"10"})
    ulx = int(geo_pos.find('ULX').text)
    uly = int(geo_pos.find('ULY').text)

    #getting the step of the image in the x and y dircetions
    xdim = int(geo_pos.find('XDIM').text)
    ydim = int(geo_pos.find('YDIM').text)

    file.close()
    #storing the metadata in a dictionary
    meta = {}
    meta['reef_name'] = fp.split('/')[1]
    meta['dt'] = dt
    meta['files'] = get_out_files(meta['reef_name'])
    meta['coords'] = coords
    meta['crs'] = crs
    meta['rows'] = rows
    meta['cols'] = cols
    meta['ulx'] = ulx
    meta['uly'] = uly
    meta['xdim'] = xdim
    meta['ydim'] = ydim

    #getting the latitude and longitude to calculate the tide at the given point
    lat = (meta['coords'][0] + meta['coords'][1])/2
    lon = (meta['coords'][2] + meta['coords'][3])/2
    #getting the tide on the day of the image
    meta['tide_level'] = tide.get_tide(lat,lon,meta['dt'])
    #getting a polygon representing the shape of the reef
    meta['polygon'] = read_gjson(meta)
    return meta

#function to read in the polygon representing the shape of the reef
def read_gjson(meta):
    #creating the filepath for the geojson file
    fp = os.path.join('data', meta['reef_name'], str(meta['reef_name']) +'.geojson')
    #loading in the geojson file into a geopandas dataframe
    df = gpd.read_file(fp)
    #setting the current crs of the dataframe
    df.crs = {'init': 'epsg:4326'}
    #changing the crs to that of the sentinel image
    df = df.to_crs(meta['crs'])
    return df

#method to read in the outfiles from depth_profile.py on a given reef
def get_out_files(reef_name):
    #opening the text file
    path = os.path.join('data',reef_name, str(reef_name) +'.txt')
    f = open(path,"r")
    #creating an empty list
    files = []
    next_line = -1
    #looping through each line in the file
    for i,l in enumerate(f):
        #if the line says output files, we want all the lines that come after it
        if l == 'Output Files:\n':
            while 1 :
                #loop till we meet an empty line or the end of the file
                line = f.readline()
                if not line or line == '\n':
                    break
                #append each filename
                files.append(line.strip())
    f.close()
    return files

#method to use the depth profiles created from ICESAT 2 data
def prep_df(fp,meta):
    #reading in the depths
    df = pd.DataFrame.from_csv(fp)
    #adjusting the heights based on the tide on the given day
    df.Height = df.Height - meta['tide_level']
    #converts the lat and lon to the crs of the image
    df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
    df['Coordinates'] = df.Coordinates.apply(Point)
    df = gpd.GeoDataFrame(df, geometry = 'Coordinates')
    df.crs = {'init': 'epsg:4326'}
    df = df.to_crs(meta['crs'])
    return df

#method to clip the larger sentinel image to a smaller area of interest
def get_bbox(coords,crs):
    #creates a bounding box for just the reef we are interested in
    ul = (coords[2], coords[1])
    br = (coords[3], coords[0])
    bbox = pd.DataFrame([ul,br], columns = ['Longitude','Latitude'])
    bbox['Coordinates'] = list(zip(bbox.Longitude, bbox.Latitude))
    bbox['Coordinates'] = bbox.Coordinates.apply(Point)
    #converts the coordinates of the bounding box to the crs of the image
    bbox = gpd.GeoDataFrame(bbox, geometry = 'Coordinates')
    bbox.crs = {'init': 'epsg:4326'}
    bbox = bbox.to_crs(crs)
    return bbox

#method that trains a regression model and returns the same
def get_regressor(base_fp):
    #gets the metadata
    fp = os.path.join(base_fp,'MTD_TL.xml')
    meta = get_metadata(fp)
    #creating a training dataset using ICESAT 2 depth profile
    reef = pd.DataFrame()
    for fn in meta['files']:
        reef = pd.concat([reef,prep_df(fn,meta)])
    #drop any nan rows
    reef = reef.dropna()
    #gets the bounding box of the reef
    bbox = get_bbox(meta['coords'], meta['crs'])
    meta['bbox'] = bbox
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
#         df['b4'] = df.Pixels.apply(lambda x: (x[2]))
        df['b8'] = df.Pixels.apply(lambda x: (x[3]))
        #converts points with pixel values above the mask to nan
        df['mask'] = df.b8.apply(lambda x: x if x < meta['mask_thresh'] else np.nan)
        return df

    print(meta['mask_thresh'])
    reef = extract_pixel_cols(reef)
    #drop all rows that are land or clouds
    reef = reef.dropna(subset = ['mask'])

    #stroes the x and y coordinates in individual columns
    reef['x'] = reef.Coordinates.x
    reef['y'] = reef.Coordinates.y
    #removing training data that are above -2m
    reef = reef.loc[reef.Height < -1.5]

    #gets just the required input columns
    x = reef.loc[:,['b2','b3']] #,'b4']]

    #subtracts the dark pixel value from each band
#     bp = (meta['black_val'])
    meta['min_pix'] = {'B02': min(x.b2), 'B03': min(x.b3)}
    bp = meta['min_pix']
    x['b2'] = x['b2'].apply(lambda x: max(1,x - bp['B02']))
    x['b3'] = x['b3'].apply(lambda x: max(1,x - bp['B03']))
    x['diff'] = x.apply(lambda x: (math.log(x['b2']) - math.log(x['b3'])), axis = 1)
#     x['b4'] = x['b4'].apply(lambda x: math.log(max(1,x - bp['B04'])))

    #gets the column that we are trying to predict
    y = reef.Height

    #performs a train test split, putting 33% of the data in the test set
    X_train, X_test, y_train, y_test = train_test_split(x.loc[:,['diff']], y, test_size=0.33, random_state =3)

    #fit a linear regression ridge model to our data
    from sklearn import linear_model
#     reg = linear_model.Ridge(alpha=.5).fit(X_train,y_train)
    reg = LinearRegression().fit(X_train,y_train)

    #predicts values for our test set
    preds = reg.predict(X_test)
    scores = cross_val_score(reg, x.loc[:,['diff']], y, cv=5)
    # print('MSE Test', mean_squared_error(y_test,preds))
    # print('MSE Train', mean_squared_error(y_train, reg.predict(X_train)))
    # print('Score Test', reg.score(X_test, y_test))
    # print('Score Train', reg.score(X_train, y_train))
    print('accuracy', np.mean(scores))
    print('var original', np.var(y_test.values))
    print(y_test.values)

    print()
    print(preds)
    print()
    print(abs(y_test.values - preds))
    print('var residual', np.var(abs(y_test.values - preds)))

    #get the different band values and height for all x and y coordinates
    out = x.copy()
    out['h'] = y
    out['x'] = reef['x']
    out['y'] = reef['y']
    meta['correlation'] = pearsonr(y_train.values, X_train.loc[:,'diff'].values)
    return reg, meta, out

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
#     b4_pix = meta['imgs'][2]
    b8_pix = meta['imgs'][3]

    #stores the masking threshold
    mask_thresh = meta['mask_thresh']
    print(mask_thresh)

    #looping through the image coordinates
    for j in range(len(b2_pix)):
        for i in range(len(b2_pix[0])):
            #getting the x and y coordinates
            x_coord = meta['bbox'].Coordinates.x[0] + ((i)* meta['xdim']) + 5
            y_coord = meta['bbox'].Coordinates.y[0] + ((j)* meta['ydim']) - 5
            p = Point((x_coord, y_coord))
            #checking if the point is within the reef
            if meta['polygon'].loc[0,'geometry'].contains(p):
                #getting the band values minus the dark pixel value
                bp = meta['black_val']
                bp = meta['min_pix']
                band_2 = max(1,b2_pix[j][i] - bp['B02'])
                band_3 = max(1,b3_pix[j][i] - bp['B03'])
    #             band_4 = math.log(max(1,b4_pix[j][i] - meta['black_val']['B04']))
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
                x_feat = [math.log(band_2) -  math.log(band_3)]

                h.append(reg.predict([x_feat])[0] + meta['tide_level'])
    #creating a dataframe with the output information and save that df as a csv
    df = pd.DataFrame([x,y,h,pix,log_pix]).T
    df.columns = ['x','y','height','pixels','normalised_pixel']
    meta['reef'] = base_fp2.split('/')[1]
    out_fp = os.path.join('data', meta['reef'], 'Output', meta['reef']+'_out_'+ meta['dt'].strftime("%Y%m%d%H%M%S") + '.csv')
    df.to_csv(out_fp)


#method to load in all the images
def get_images(base_fp,meta):
    img_time = base_fp.split('/')[2].split('_')[2]
    img_start = base_fp.split('/')[2].split('_')[-2]
    #select the bands that we want
    bands = ['B02','B03','B04','B08']
    imgs = []
    black = {}
    #loops through the bands
    for b in bands:
        #creates file path
        img_path = os.path.join(base_fp, 'IMG_DATA', (img_start + '_' + img_time + '_' + b+ '.jp2'))
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

        #black pixel value is not being used in our code right now, if we do use it we will need to change the coordinates
        #clip a dark pixel corner in the image
        dark = [x[:500] for x in img[1098:]]
        #use the median as the dark pixel value
        black_val = np.median(dark)
        black[b] = black_val


        imgs.append(img)
    #store the dark pixel value in the metadata
    meta['black_val'] = black
    #return the images
    return imgs

#method to plot the reef depth histogram and a scatter plot of the same
def plot_reefs(df,data):
    #plot histogram of depths
    fig, ax = plt.subplots(figsize = (7,7))
    df['height'].plot.hist(bins = np.arange(math.floor(df.height.min()),math.ceil(df.height.max()),1))
    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Depth Histogram')

    #plot scatter plot
    fig,ax = plt.subplots(figsize = (7,7))
    #getting just depths between -25m to 10m
    df = df.loc[(df.height <= 10) & (df.height >= -25)]
    #creating a color scale at 5m intervals
    print(df.height.max(), df.height.min())
    cmap = cm.colors.ListedColormap(['black','navy','mediumblue' ,'blue','royalblue', 'dodgerblue',
                                     'skyblue','limegreen',  'lime' , 'yellow'
                                      ,'orange','tomato',
                                     'red','firebrick' ,'maroon'])
    bounds = np.arange(-25,11,2.5)
#     else:
#         cmap = cm.colors.ListedColormap(['black','navy',  'blue','royalblue', 'dodgerblue', 'cornflowerblue',
#                                        'teal',  'aquamarine', 'mediumspringgreen' , 'orange','darkorange',
#                                          'coral','tomato','orangered','red','firebrick' ,'darkred','maroon'])
#         bounds = np.arange(-45,50,5)
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
    pts = ax.scatter(x = df.x, y = df.y, c = df.height, s= 1, cmap = cmap, norm = norm)
    #scatter plot of the track lines from the ICESAT 2 data
    ax.scatter(x = data.x, y= data.y, s = 3, c = 'black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Depth map in meters')

    return

#method to see the relationship between the input variable of our model and the predicted variable
def band_plots(data):
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    #gets difference between consecutive x values, allowing us to recognise if two tracks are separated
    diffs = np.diff(data.x)
    beams = []
    beam = 1
    for x in abs(diffs):
        beams.append(beam)
        #if the difference is more than 2000, we are working with the next beam
        if x > 2000:
            beam += 1
    beams.append(beam)
    data['beam'] = beams

    #changing column names
    data['log(band2) - log(band3)'] = data['diff']
    data['Height (m)'] = data.h

    #creating jointplot of our data
    g0 = sns.jointplot(data['Height (m)'], data['log(band2) - log(band3)'])
    g0.ax_joint.scatter(data['Height (m)'],data['log(band2) - log(band3)'])

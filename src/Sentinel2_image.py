import os
from pathlib import Path
import glob
from bs4 import BeautifulSoup
from datetime import datetime
import rasterio
import numpy as np
import geopandas as gpd

import src.Tide_API as tide

class Sentinel2_image():
    def __init__(self,safe_file_path,coords):
        self.safe_file_path = safe_file_path
        self.reef_path = os.path.dirname(safe_file_path)
        self.safe_file = os.path.basename(safe_file_path)
        self.reef_name = os.path.basename(self.reef_path)
        self.bbox_coords = coords

        self.predictions_path = os.path.join(self.reef_path,'Output', 'Depth_Predictions')
        self.imgs_path = os.path.join(self.predictions_path, 'Imgs')
        self.depth_preds_path = os.path.join(self.predictions_path, 'Reef_depth_predictions')
        self.training_data_path = os.path.join(self.predictions_path,'Training_data')
        self.dirs = [self.predictions_path, self.imgs_path, self.depth_preds_path,\
                        self.training_data_path]

        fn = 'MTD_TL.xml'
        self.meta_path = list(Path(self.reef_path).glob('**/' + self.safe_file + '/**/' + fn))[0]
        self.meta = self.get_metadata()

        self.tide_level = tide.get_tide(coords,self.get_date())


    def create_directories(self):
        for dir in self.dirs:
            if not os.path.exists(dir):
                os.mkdir(dir)

    def get_coords(self):
        return self.bbox_coords

    def get_tide(self):
        return self.tide_level

    def get_safe_file(self):
        return self.safe_file

    def get_img_path(self):
        return self.imgs_path

    def get_file_directories(self):
        return [self.imgs_path, self.depth_preds_path, self.training_data_path]

    def get_reef_name(self):
        return self.reef_name

    def get_metadata(self):
        metadata_file = open(self.meta_path, 'r')
        contents = metadata_file.read()
        soup = BeautifulSoup(contents,'xml')
        meta = {}
        #getting the time of the image and creating a datetime object
        dt = (soup.find('SENSING_TIME').text.split('.')[0].replace('T',''))
        meta['dt'] = datetime.strptime(dt, '%Y-%m-%d%H:%M:%S')

        #getting the crs of the image
        geo_info = soup.find('n1:Geometric_Info')
        meta['crs'] = geo_info.find('HORIZONTAL_CS_CODE').text.lower()

        geo_pos = geo_info.find('Geoposition' , {'resolution':"10"})
        #getting the step of the image in the x and y dircetions
        meta['xdim'] = int(geo_pos.find('XDIM').text)
        meta['ydim'] = int(geo_pos.find('YDIM').text)

        metadata_file.close()
        return meta


    def get_date(self):
        if 'dt' in self.meta:
            return self.meta['dt']
        else:
            return None

    def get_crs(self):
        return self.meta['crs']

    #function to read in the polygon representing the shape of the reef
    def read_gjson(self):
        #creating the filepath for the geojson file
        fp = os.path.join(self.reef_path, os.path.basename(self.reef_path) +'.geojson')
        #loading in the geojson file into a geopandas dataframe
        df = gpd.read_file(fp)
        #setting the current crs of the dataframe
        df.crs = {'init': 'epsg:4326'}
        #changing the crs to that of the sentinel image
        df = df.to_crs(self.meta['crs'])
        return df

    def get_meta(self):
        return self.meta

    #method to load in all the images
    def load_sentinel(self):
        #select the bands that we want
        bands = ['B02','B03','B04','B08']
        imgs = []
        m = []
        from rasterio import mask
        geom = self.read_gjson()['geometry']
        bb = geom.bounds
        self.meta['ulx'] = geom.bounds.minx[0]
        self.meta['uly'] = geom.bounds.maxy[0]

        #loops through the bands
        for b in bands:
            img_dir = os.path.dirname(self.meta_path)
            img_path = list(Path(img_dir).glob('**/' + 'IMG_DATA' + '/**/*'+b+'_10m.jp2'))[0]
            #reads in image
            band = rasterio.open(img_path, driver = 'JP2OpenJPEG')
            out_image, out_transform = mask.mask(band, geom, crop=True)
            imgs.append(out_image)
        self.meta['imgs'] = imgs
        return imgs

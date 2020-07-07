import re
from datetime import datetime
import h5py
import os
import json
import src.Tide_API as tide
import numpy as np

class IS2_file():

    def __init__(self,h5_dir,h5_fn,bbox_coordinates):
        self.h5_dir = h5_dir
        self.h5_fn = h5_fn
        self.h5_file = os.path.join(self.h5_dir, self.h5_fn)
        self.bbox_coordinates = bbox_coordinates
        self.sc_orient = self.get_orientation()
        if self.sc_orient == 1:
            self.strong_lasers = ['gt1r', 'gt2r', 'gt3r']
        else:
            self.strong_lasers = ['gt1l', 'gt2l', 'gt3l']

        metadata = self.load_json()
        metadata[self.h5_fn] = {}
        if 'tide' in metadata[self.h5_fn]:
            self.tide_level = metadata[self.h5_fn]['tide']
        else:
            self.tide_level = tide.get_tide(self.bbox_coordinates, self.get_date())
            metadata[self.h5_fn]['tide'] = self.tide_level
        self.metadata = metadata
        self.sea_level_func = {}

    def set_sea_level_function(self,sea,laser):
        self.sea_level_func[laser] = list(sea)
        self.metadata[self.h5_fn]['sea_level_func'] = self.sea_level_func
        self.write_json(self.metadata)


    def get_sea_level_function(self,laser):
        return np.poly1d(self.load_json()[self.h5_fn]['sea_level_func'][laser])
        # return np.poly1d(self.sea_level_func[laser])

    def get_strong_lasers(self):
        return self.strong_lasers

    def get_fn(self):
        return self.h5_fn

    def get_file_tag(self):
        return self.h5_fn.split('.')[0]

    def get_bbox_coordinates(self):
        return self.bbox_coordinates

    def get_tide(self):
        return self.tide_level

    def load_json(self):
        reef_path = os.path.dirname(self.h5_dir)
        metadata_path = os.path.join(reef_path, 'ICESAT_metadata.json')
        if os.path.exists(metadata_path):
            metadata = json.load(open(metadata_path))
            return metadata
        else:
            return {}

    def write_json(self,d):
        reef_path = os.path.dirname(self.h5_dir)
        metadata_path = os.path.join(reef_path, 'ICESAT_metadata.json')
        with open(metadata_path, 'w') as f:
            f.write(json.dumps(d, sort_keys=True, indent=4, separators=(',', ': ')))

    #method to extract the date from the filename
    def get_date(self):
    	#finding string sequence with 14 consecutive integers
        dt = re.findall('\d{14}',self.h5_fn)[0]
        #convert string to datetime object
        dt = datetime.strptime(dt, '%Y%m%d%H%M%S')
        return dt

    def get_track(self):
        track_string = re.findall('_\d{8}_',self.h5_fn)[0]
        track = track_string[1:5]
        return track

    def get_orientation(self):
        h5 = h5py.File(self.h5_file,'r')
        sc_orient = h5['orbit_info']['sc_orient'][...][0]
        return sc_orient

    def get_photon_data(self, laser):
        h5 = h5py.File(self.h5_file,'r')
        photon_data = h5[laser]['heights']
        height = photon_data['h_ph'][...]
        lat = photon_data['lat_ph'][...]
        lon = photon_data['lon_ph'][...]
        conf = photon_data['signal_conf_ph'][...]
        return [height,lat,lon,conf]

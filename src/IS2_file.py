import re
from datetime import datetime
import h5py
import os
import json
import numpy as np

import src.Tide_API as tide

class IS2_file():
    """
    Class to represent an ICESAT-2 file
    """

    def __init__(self,h5_dir,h5_fn,bbox_coordinates):
        """
        Initialise ICESAT-2 file object
        Params - 1. h5_dir (str) - directory of h5 file
                 2. h5_fn (str) - name of h5 file
                 3. bbox_coordinates [min-x, min-y, max-x, max-y] - coordinates of bounding box of coral reef.
        """
        #setting input variables as class variables
        self.h5_dir = h5_dir
        self.h5_fn = h5_fn
        self.bbox_coordinates = bbox_coordinates
        #creating h5 file path
        self.h5_file = os.path.join(self.h5_dir, self.h5_fn)
        #getting orientation of satellite and choosing strong lasers accordingly
        self.sc_orient = self.get_orientation()
        if self.sc_orient == 1:
            self.strong_lasers = ['gt1r', 'gt2r', 'gt3r']
        else:
            self.strong_lasers = ['gt1l', 'gt2l', 'gt3l']
        #load in json with metadata
        metadata = self.load_json()
        #create new entry for current h5 file
        metadata[self.h5_fn] = {}
        #load in tide or calculate it if it doesnt exist
        if 'tide' in metadata[self.h5_fn]:
            self.tide_level = metadata[self.h5_fn]['tide']
        else:
            self.tide_level = tide.get_tide(self.bbox_coordinates, self.get_date())
            metadata[self.h5_fn]['tide'] = self.tide_level
        self.metadata = metadata
        self.sea_level_func = {}

    def set_sea_level_function(self,sea,laser):
        """
        Params - 1. sea (np.poly1d) - sea level equation for given laser
                 2. laser (str) - laser of ICESAT we are interested in
        Saves sea level function in metadata
        """
        #storing sea level in metadata
        self.sea_level_func[laser] = list(sea)
        self.metadata[self.h5_fn]['sea_level_func'] = self.sea_level_func
        #saving metadata to json
        self.write_json(self.metadata)


    def get_sea_level_function(self,laser):
        """
        1. Param - laser (str) - laser of ICESAT we are interested in
        Return - np.poly1d - equation of line that represents the sea level
        """
        return np.poly1d(self.load_json()[self.h5_fn]['sea_level_func'][laser])
        # return np.poly1d(self.sea_level_func[laser])

    def get_strong_lasers(self):
        """
        Return - list - strong lasers depending on sc-orientation
        """
        return self.strong_lasers

    def get_fn(self):
        """
        Return - str - filename with .h5 extension
        """
        return self.h5_fn

    def get_file_tag(self):
        """
        Return - str - filename without .h5 extension
        """
        return self.h5_fn.split('.')[0]

    def get_bbox_coordinates(self):
        """
        Return - [min-x, min-y, max-x, max-y] - coordinates of bounding box of coral reef.
        """
        return self.bbox_coordinates

    def get_tide(self):
        """
        Return - int - tide level on date of orbit
        """
        return self.tide_level

    def load_json(self):
        """
        Loads in metadata of all the reef's ICESAT-2 files
        """
        reef_path = os.path.dirname(self.h5_dir)
        metadata_path = os.path.join(reef_path, 'ICESAT_metadata.json')
        if os.path.exists(metadata_path):
            metadata = json.load(open(metadata_path))
            return metadata
        else:
            return {}

    def write_json(self,d):
        """
        Outputs a file containing the metadata of all ICESAT-2 files
        Params - 1. d (dict) - dictionary containing metadata
        """
        reef_path = os.path.dirname(self.h5_dir)
        metadata_path = os.path.join(reef_path, 'ICESAT_metadata.json')
        with open(metadata_path, 'w') as f:
            f.write(json.dumps(d, sort_keys=True, indent=4, separators=(',', ': ')))

    def get_date(self):
        """
        Extract date from ICESAT file name
        Return - datetime - datetime of orbit
        """
    	#finding string sequence with 14 consecutive integers
        dt = re.findall('\d{14}',self.h5_fn)[0]
        #convert string to datetime object
        dt = datetime.strptime(dt, '%Y%m%d%H%M%S')
        return dt

    def get_track(self):
        """
        Extracting track of laser using ICESAT file name
        Return - str - track of ICESAT orbit
        """
        #finding string with track and cycle
        track_string = re.findall('_\d{8}_',self.h5_fn)[0]
        #returning ICESAT track
        track = track_string[1:5]
        return track

    def get_orientation(self):
        """
        Extract the orientation of the satelitte
        Return - int - sc orientation
        """
        #loads in h5 file and extracts the sc orientation
        h5 = h5py.File(self.h5_file,'r')
        sc_orient = h5['orbit_info']['sc_orient'][...][0]
        return sc_orient

    def get_photon_data(self, laser):
        """
        Extract the required photon information from ICESAT-2 data output
        Return - list - required photon data output
        """
        #load in h5 file
        h5 = h5py.File(self.h5_file,'r')
        #returns list of photon height, lat,lon and photon confidence
        photon_data = h5[laser]['heights']
        height = photon_data['h_ph'][...]
        lat = photon_data['lat_ph'][...]
        lon = photon_data['lon_ph'][...]
        conf = photon_data['signal_conf_ph'][...]
        return [height,lat,lon,conf]

    def get_tide_ib(self, laser):
        """
        Extract the required photon information from ICESAT-2 data output
        Return - list - required photon data output
        """
        #load in h5 file
        h5 = h5py.File(self.h5_file,'r')
        #returns list of photon height, lat,lon and photon confidence
        photon_data = h5[laser]['geophys_corr']
        tide = photon_data['tide_ocean'][...]
        ib = photon_data['dac'][...]
        return [tide,ib]

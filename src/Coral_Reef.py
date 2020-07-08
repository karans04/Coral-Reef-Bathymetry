import os
import geopandas as gpd


class Coral_Reef():
    """
    Class to represent a coral reef
    """
    def __init__(self, data_dir, reef_name):
        """
        Initialise a coral reef object
        Params 1. data_dir (str) - directory to save outfiles
               2. reef_name (str) - name of the coral reef
        """
        #stores input variables as class variables
        self.data_dir = data_dir
        self.reef_name = reef_name
        #path for outfiles
        self.reef_path = os.path.join(self.data_dir, self.reef_name)
        #stores coordinates of bounding box
        self.bbox_coords = self.get_bounding_box()
        
        #creating directories to save data output
        self.outfile_path = os.path.join(self.reef_path, 'Output')
        self.data_cleaning_path = os.path.join(self.outfile_path, 'Data_Cleaning')
        self.icesat_path = os.path.join(self.data_cleaning_path, 'ICESAT_photons')
        self.proc_path = os.path.join(self.data_cleaning_path, 'Processed_output')
        self.images_path = os.path.join(self.data_cleaning_path, 'Imgs')
        self.data_plots_path = os.path.join(self.data_cleaning_path,'Data_plots')
        self.dirs = [self.reef_path, self.outfile_path, self.data_cleaning_path, \
                        self.icesat_path, self.proc_path, self.images_path,self.data_plots_path]
        self.create_directories()

    def get_reef_name(self):
        """
        Return - str - name of coral reef 
        """
        return self.reef_name

    def get_path(self):
        """
        Return - str - path of the reef in data directory
        """
        return self.reef_path

    def get_outpath(self):
        """
        Return str - path of output directory
        """
        return self.outfile_path

    def create_directories(self):
        """
        Creates directories for all outfiles
        """
        #checks if directory exists, if not it is created
        for dir in self.dirs:
            if not os.path.exists(dir):
                os.mkdir(dir)

    def get_file_drectories(self):
        """
        Return - [output paths] - paths 
        """
        return [self.icesat_path,self.proc_path,self.images_path,self.data_plots_path]

    def get_processed_output_path(self):
        """
        Return - str - path for processed ICESAT output
        """
        return self.proc_path

    def get_bounding_box(self):
        """
        Gets the coordinates of bounding box around coral reef
        Return - [min-x min-y max-x max-y]
        """
        #loads in geojson of reef into geopandasa
        geojson_fp = os.path.join(self.reef_path, self.reef_name + '.geojson')
        reef_gjson = gpd.read_file(geojson_fp)
        #returns coordinates of bounding box around coral reef
        reef_polygon = reef_gjson.geometry[0]
        coords = reef_polygon.bounds
        return coords

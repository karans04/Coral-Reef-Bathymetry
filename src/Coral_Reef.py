import os
import geopandas as gpd


class Coral_Reef():

    def __init__(self, data_dir, reef_name):
        self.data_dir = data_dir
        self.reef_name = reef_name
        self.reef_path = os.path.join(self.data_dir, self.reef_name)
        self.bbox_coords = self.get_bounding_box()
        self.file_directories = []
        self.outfile_path = os.path.join(self.reef_path, 'Output')
        self.data_cleaning_path = os.path.join(self.outfile_path, 'Data_Cleaning')
        self.icesat_path = os.path.join(self.data_cleaning_path, 'ICESAT_photons')
        self.proc_path = os.path.join(self.data_cleaning_path, 'Processed_output')
        self.images_path = os.path.join(self.data_cleaning_path, 'Imgs')
        self.data_plots_path = os.path.join(self.data_cleaning_path,'Data_plots')
        self.dirs = [self.reef_path, self.outfile_path, self.data_cleaning_path, \
                        self.icesat_path, self.proc_path, self.images_path,self.data_plots_path]

    def get_reef_name(self):
        return self.reef_name

    def get_path(self):
        return self.reef_path

    def get_outpath(self):
        return self.outfile_path

    def create_directories(self):
        for dir in self.dirs:
            if not os.path.exists(dir):
                os.mkdir(dir)

    def get_file_drectories(self):
        return [self.icesat_path,self.proc_path,self.images_path,self.data_plots_path]

    def get_processed_output_path(self):
        return self.proc_path

    def get_bounding_box(self):
        geojson_fp = os.path.join(self.reef_path, self.reef_name + '.geojson')
        reef_gjson = gpd.read_file(geojson_fp)
        reef_polygon = reef_gjson.geometry[0]
        coords = reef_polygon.bounds
        #min-x min-y max-x max-y
        return coords

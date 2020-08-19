Coral_Reef.py is a class used to represent a coral reef. 

Requirements - create a folder in the data directory named after the reef. The folder must contain a geojson file with a POLYGON of the reef. The geojson must be named after the reef -> reef_name.geojson

To create a coral reef object: </br>
import src.Coral_Reef as coral_reef </br>
reef = coral_reef.Coral_Reef(data_dir, reef_name)</br>

Methods that a reef object can call: </br>
1. reef.get_reef_name() - returns the name of the coral reef</br>
2. reef.get_path() - returns the path of the reef folder (data_dir/reef_name)</br>
3. reef.get_outpath() - returns the path of the output file </br>
4. reef.create_directories() - does not return anything. Just creates the directories for the data output</br>
5. reef.get_file_directories() - returns the file paths of directories contained in Data_Cleaning </br>
6. reef.get_processed_outpath_file() - returns the path for processed ICESAT output </br>
7. reef.get_bounding_box() - returns the coordinates of a bounding box around the reef, with the given format: </br>
[min-x min-y max-x max-y] 
<br/> x = longitude 
<br/> y = latitude 
</br></br>Files created from reef.get_file_directories()


```
data
├── reef_name
    ├── reef_name.geojson
    ├── Output
        ├── Data_Cleaning
            ├── ICESAT_photons - raw photons
            ├── Processed_output - photons after depth prediction
            ├── Imgs - plots of the depth predictions
            └── Data_plots - csv containing raw photons and depth predictions (used for plots)
```

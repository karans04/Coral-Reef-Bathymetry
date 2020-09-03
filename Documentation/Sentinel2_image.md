Sentinel2_image is a class used to represent a Sentinel SAFE file. </br>

Requirements - create a folder in the data directory named after the reef. The folder must contain a geojson file with a POLYGON of the reef. The geojson must be named after the reef -> reef_name.geojson

To create a Sentinel2_image object:
import src.Sentinel2_image as sentinel
reef = sentinel.Sentinel2_image(safe_file_path, coords)

1. create_directories() - creates directories for the data output. Can be seen below in the data output format. </br>
2. get_coords() - returns the coordiantes of the reef.
3. get_tide() - returns the tide over the reef on the day of the Sentinel image.
4. get_safe_file() - returns the name of the sentinel image as a string.
5. get_img_path() - returns the path in the data output where images are saved.
6. get_file_directories() - returns a list of the paths for the data output.
7. get_reef_name() - returns the name of the reef as a string. 
8. get_metadata() - loads in required metadata from sentinel product (ulx, uly,crs etc).
9. get_date() - returns the date of the sentinel2 image. 
10. get_crs() - returns the coordinate system of the sentinel image. 
11. read_gjson() - reads in the geojson file of the coral reef. 
12. get_meta() - returns a dictionary containing the metadata of the sentinel image. 
13. load_sentinel() - returns a list of the pixel values for the band 2,3,4 and 8.
14. get_tci() - returns a true colour image of the sentinel image. 

Data Output format
```
data
├── reef_name
    ├── reef_name.geojson
    ├── Output
        ├── Depth_Predictions
            ├── Training_data - photon pixel value vs depth from ICESAT 2
            ├── Reef_depth_predictions - predictions for the rest of the reef
            ├── Imgs - plots of the depth predictions
```

IS2_file.py is a class used to represent a ICESAT-2 file. 

Requirements - Download IS2 files using ATL03_API.py. H5 files downloaded will saved in a directory called H5 that is in the reef directory. 

To create a IS2_file object: </br>
import src.IS2_file as is2 </br>
reef = is2.IS2_file(h5_dir, h5_filename, bounding_box_coords)</br>

Methods that a reef object can call: </br>
1. is2.set_sea_level_function(sea,laser) - saves np.poly1d sea level equation to metadata for the passed in laser string. 
2. is2.get_sea_level_function(laser) - returns a np.poly1d equation of line that represents the sea level for the passed in laser string.
3. is2.get_strong_lasers() - returns a list containing strings that represent the strong lasers on the current orientation.
4. is2.get_fn() - returns a string with the filename including the .h5 extension.
5. is2.get_file_tag() - returns a string with the filename excluding the .h5 extension.
6. is2.get_tide() - returns int of the tide on the day the satellite is orbiting. 
7. is2.load_json() - loads in metadata json that is stored in the reef directory. Metadata json contains the sea level function and tide for each IS2 file.
8. is2.write_json(d) - writes metadata from dictionary d to ICESAT_metadata.json.
9. is2.get_date() - returns a datetime object representing the date of the current passover. 
10. is2.get_track() - returns a string containing the ground track of the current passover. 
11. is2.get_orientation() - returns a int representing the current orientation of the satellite (0 or 1 - helps determine the strong lasers) 
12. is2.get_photon_data(laser) - returns a list [height,lat,lon,conf] containing photon data from the h5 file for the passed in laser string.
13. is2.get_bounding_box() - returns the coordinates of a bounding box around the reef, with the given format: </br>
[min-x min-y max-x max-y] 
<br/> x = longitude 
<br/> y = latitude 



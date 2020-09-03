Setup 

In working directory, there should be a folder called data containing a subfolder for each reef being analyzed. Within each subfolder should be the following folders/files:
</br>
<blockquote>
- a GeoJSON file containing the outline of the reef, named reef_name.geojson. This file is obtained from www.geojson.io/. Switch to "OSM model" in bottom left corner of window, create a polygon of reef with the cursor, save points using the menu at upper left (save->GeoJSON), then rename map.geojson file and move to data directory.
</br></br>
Run depth_profile.py for the reef. Output will be saved in data/reef_name/Output/Data_Cleaning/Processed_output
</blockquote> </br></br>
Functions 


Functions </br>
1. prep_df(sf, fp,crs) 
</br>
2. load_ICESAT_predictions(icesat_proc_path, sf)
</br>
3. get_regressor(reef, sf)
</br>
4. get_pixel_val(coord)
</br>
5. extract_pixel_cols(df)
</br>
6. remove_log_outliers(data)
</br>
7. predict_reef(ref, sf, master_df)
</br>
8. all_safe_file(reef)

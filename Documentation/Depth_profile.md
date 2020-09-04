Setup </br>

In working directory, there should be a folder called data containing a subfolder for each reef being analyzed. Within each subfolder should be the following folders/files:
</br>
<blockquote>
- a GeoJSON file containing the outline of the reef, named reef_name.geojson. This file is obtained from www.geojson.io/. Switch to "OSM model" in bottom left corner of window, create a polygon of reef with the cursor, save points using the menu at upper left (save->GeoJSON), then rename map.geojson file and move to data directory.</br></br>
- a folder named H5 which contains ICESat-2 ATL03 data files for this reef. These are HDF5 files that can be obtained from OpenAltimetry ( http://www.openaltimetry.org) or from NASA EarthData search (https://search.earthdata.nasa.gov/search/granules?p=C1705401930-NSIDC_ECS). </br>
Files could also be downloaded using https://github.com/karans04/Coral-Reef-Bathymetry/blob/master/src/ATL03_API.py. 
</blockquote> </br></br>
Work flow </br></br>

![image info](./assets/Depth_profile.png)

Functions 

1. create_photon_df( photon_data - [Height,Latitude,Longitude,Confidence] ) - data obtained from h5 file. </br>
Return - pd.DataFrame </br>

2. individual_confidence( df (pd.DataFrame) ) - helper function that converts the confidence array into individual confidence scores for land, ocean, sea ice, land ice and inland water. Adds each confidence score as a new column to the pandas dataframe passed in. </br>
Return - pd.DataFrame </br>

3. convert_h5_to_csv( is2_file (IS2_file object), laser (str) , out_fp (str) ) - helper function that saves a csv at out_fp (fielpath) for the passed in laser of the h5 file.
Return - pd.DataFrame </br>

4. apply_DBSCAN( df (pd.DataFrame), out_path (str), is2 (IS2_file object), laser(str) ) - apply DBSCAN algorithm on photon data to cluster between reef and noise. Adds a column called labels that determines if the photon is reef or noise. Labels >= 0 are reef photons.
Return - pd.DataFrame </br>

5. depth_profile_adaptive( df (pd.DataFrame), out_path (str), is2 (IS2_file object),laser(str) ) - rolling window (dx = 0.0005) - Gets median of all photons in window. If no photons in window the window size increases by dx. </br>
Return - pd.DataFrame </br>

6. remove_depth_noise( depths ([depth_predictions]) ) - retains depth predictions at index i, if i-1 and i+1 are not nan. Only required if depth_profile_adaptive is used instead of DBSCAN. </br>
Return [depth_predictions] </br>

7. combine_is2_reef( is2 (IS2_file object),depths (pd.DataFrame) ) - combines depth_predictions from func6 with photon data. Only required if depth_profile_adaptive is used instead of DBSCAN.  </br>
Returns - pd.DataFrame </br>

8. prcoess_h5( reef (Coral Reef object), is2_file (IS2_file object) ) - clusters reef photons from noise for ICESAT2 file. This process is only done for the strong lasers. </br>

9. get_depths( reef (Coral Reef object) ) - controller function running depth profile for all .h5 files downloaded. </br>

</br>
Sample Output </br>

![image](./../reefs/Moce_IS2_sample1.png)

![image](./../reefs/Moce_IS2_sample2.png)

![image](./../reefs/Ono_i_Lau_IS2_sample1.png)

![image](./../reefs/Ono_i_Lau_IS2_sample2.png)

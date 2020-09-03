Setup </br>

In working directory, there should be a folder called data containing a subfolder for each reef being analyzed. Within each subfolder should be the following folders/files:
</br>
<blockquote>
- a GeoJSON file containing the outline of the reef, named reef_name.geojson. This file is obtained from www.geojson.io/. Switch to "OSM model" in bottom left corner of window, create a polygon of reef with the cursor, save points using the menu at upper left (save->GeoJSON), then rename map.geojson file and move to data directory.</br>
- a folder named H5 which contains ICESat-2 ATL03 data files for this reef. These are HDF5 files that can be obtained from OpenAltimetry ( http://www.openaltimetry.org) or from NASA EarthData search (https://search.earthdata.nasa.gov/search/granules?p=C1705401930-NSIDC_ECS)
</blockquote>

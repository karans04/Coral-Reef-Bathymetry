Setup </br>

In working directory, there should be a folder called data containing a subfolder for each reef being analyzed. Within each subfolder should be the following folders/files:
</br>
<blockquote>
- a GeoJSON file containing the outline of the reef, named reef_name.geojson. This file is obtained from <a href = "www.geojson.io/"> </a>. Switch to "OSM model" in bottom left corner of window, create a polygon of reef with the cursor, save points using the menu at upper left (save->GeoJSON), then rename map.geojson file and move to data directory.</br>
- a folder named H5 which contains ICESat-2 ATL03 data files for this reef. These are HDF5 files that can be obtained from OpenAltimetry (<a href = "http://www.openaltimetry.org"> </a>) or from NASA EarthData search (<a href = "https://search.earthdata.nasa.gov/search/granules?p=C1705401930-NSIDC_ECS"> </a>)
</blockquote>

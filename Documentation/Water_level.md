Functions
1. get_water_level( df (pd.DataFrame) ) - fit line to water level </br>
Return np.poly1d - line representing sea level  </br>
2. adjust_for_speed_of_light_in_water( df (pd.DataFrame) , tide_level (int) ) - recalibrate the depths taking into consideration the tide as well as the change in the speed of light from air to water. </br>
Return pd.DataFrame - with adjusted depths </br>
3. adjust_for_refractive_index( df (pd.DataFrame) ) - recalibrate depths due to change in refractve index from air to water. </br>
Return pd.DataFrame - with adjusted depths </br>
4. normalise_sea_level( df (pd.DataFrame) ) - adjust all depths to mean water level. </br>
Return pd.DataFrame - with adjusted depths </br>
np.poly1d - line representing sea level

import matplotlib.pyplot as plt
import pandas as pd
import src.Water_level as water_level
import numpy as np

import src.IS2_file as is2File

def p_is2(m,is2,laser,images_fp):


    #readjusting photon depths to the sea level
    f = is2.get_sea_level_function(laser)
    # print(f)
    lats = m.Latitude.drop_duplicates()
    sea = f(lats)
    mean_sea = np.mean(sea)
    sea = sea - np.mean(sea)

    #thresholds to classify photon as land or reef or water
    threshold = -0.5
    threshold_land = 2
    #plotting the sea, land and reef in different colours based on the thresholds
    sea_level = m.loc[(m.Height > threshold) & (m.Height < threshold_land) ]
    reef = m.loc[m.labels > 0]
    noise = m.loc[m.labels == -1]
    land = m.loc[m.Height > threshold_land]
    fig,ax = plt.subplots()
    plt.scatter(sea_level.Latitude, sea_level.Height, s = 0.1)
    plt.scatter(reef.Latitude, reef.Height, s= 0.1, color = 'green')
    plt.scatter(land.Latitude, land.Height, s =0.1, color = 'brown')
    plt.scatter(noise.Latitude, noise.Height, s= 0.1, color = 'blue')
    plt.xlabel('Latitude')
    plt.ylabel('Height (m)')
    date = is2.get_date()
    track = is2.get_track()
    plt.title('Track: ' + str(track) + ' - Date: ' + str(date.date()) + ' - Laser: ' + laser)

    plt.plot(lats, sea, linestyle='-', marker='o',color='blue', linewidth=3, markersize=2)
    # plt.scatter(reef_ph.Latitude, reef_ph.Height, s=0.1, color = 'green')

    # depth_prof = plt.plot(noise.Latitude, noise.Height, linestyle='-', marker='o',color='orange', linewidth=1, markersize=2,alpha = 0.4)
    plt.savefig(images_fp + '/photon_preds' + str(date.date())+laser + '.png')
    return ax


# def p_is2(m,is2,laser,images_fp):
#
#
#     #readjusting photon depths to the sea level
#     f = is2.get_sea_level_function(laser)
#     # print(f)
#     lats = m.Latitude.drop_duplicates()
#     sea = f(lats)
#     mean_sea = np.mean(sea)
#     sea = sea - np.mean(sea)
#
#     #thresholds to classify photon as land or reef or water
#     threshold = -0.3
#     threshold_land = 2
#     #plotting the sea, land and reef in different colours based on the thresholds
#     sea_level = m.loc[(m.Photon_depth > threshold) & (m.Photon_depth < threshold_land) ]
#     reef = m.loc[m.Photon_depth <= threshold]
#     land = m.loc[m.Photon_depth > threshold_land]
#     fig,ax = plt.subplots()
#     plt.scatter(sea_level.Latitude, sea_level.Photon_depth, s = 0.1)
#     plt.scatter(reef.Latitude, reef.Photon_depth, s= 0.1, color = 'green')
#     plt.scatter(land.Latitude, land.Photon_depth, s =0.1, color = 'brown')
#     plt.xlabel('Latitude')
#     plt.ylabel('Height (m)')
#     date = is2.get_date()
#     track = is2.get_track()
#     plt.title('Track: ' + str(track) + ' - Date: ' + str(date.date()) + ' - Laser: ' + laser)
#
#     #plotting depth predictions on top of the photon data
#     # print(sea)
#     # print(len(sea))
#     # print(lats)
#     # print(len(lats))
#     # print(mean_sea)
#     plt.plot(lats, sea, linestyle='-', marker='o',color='blue', linewidth=3, markersize=2)
#     depth_prof = plt.plot(m.Latitude, m.Predicted_depth, linestyle='-', marker='o',color='orange', linewidth=1, markersize=2,alpha = 0.4)
#     plt.savefig(images_fp + '/photon_preds' + str(date.date())+laser + '.png')
#     return ax

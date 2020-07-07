#plotting packages
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import patches

import pandas as pd
import numpy as np
import os

def plot_median_variance_graph(medians, variances, median_threshold, variance_threshold,outpath):
    ax=sns.scatterplot(x = medians, y = variances, legend = 'full')
    plt.xlabel('Median pixel value for log difference')
    plt.ylabel('Variance pixel value for log difference')

    ax.add_patch(
        patches.Rectangle(
            xy=(-median_threshold, -variance_threshold),  # point of origin.
            width=2*median_threshold,
            height=2*variance_threshold,
            linewidth=1,
            color='red',
            fill=False
        )
    )

    fn = os.path.join(outpath,'median_vs_variance.png')
    plt.savefig(fn)

#method to plot the reef depth histogram and a scatter plot of the same
def plot_reefs(fp,data,sf,line):
    df = pd.read_csv(fp)
    #plot histogram of depths
    fig, ax = plt.subplots(1,3,figsize = (28,12))
    df['Height'].plot.hist(bins = np.arange(-30,20,1), ax = ax[0])
    ax[0].set_xlabel('Height (m)')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title(sf.get_reef_name() + ' Depth Histogram')

    #getting just depths between +- 45m
    df = df.loc[(df.Height <= 10) & (df.Height >= -25)]
    #creating a color scale at 5m intervals

    cmap = cm.colors.ListedColormap(['black','navy','mediumblue' ,'blue','royalblue', 'dodgerblue',
                                     'skyblue','limegreen',  'lime' , 'yellow'
                                      ,'orange','tomato',
                                     'red','firebrick' ,'maroon'])
    bounds = np.arange(-25,11,2.5)

    norm = BoundaryNorm(bounds,cmap.N)

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
        spacing='proportional', ticks=bounds, boundaries=bounds, format='%.1f')


    #scatter plot of the predicted depths
    pts = ax[2].scatter(x = df.x, y = df.y, c = df.Height, s= 1, cmap = cmap, norm = norm)
    #scatter plot of the track lines from the ICESAT 2 data
    ax[2].scatter(x = data.x, y= data.y, s = 3, c = 'black', label = 'ICESAT-2 tracks')
    custom_lines = [Line2D([0], [0], color='black', lw=4)]
    ax[2].legend(custom_lines, ['ICESAT-2 tracks'])
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    ax[2].set_title(sf.get_reef_name() + ' Depth Predictions (m)')


    r = 3
    sns.scatterplot(x = data['diff'], y = data.Height, color = 'blue', ax = ax[1])
    sns.lineplot(x = [-r,r], y = [line(-r),line(r)], color = 'black', ax = ax[1])
    ax[1].set_xlabel('Log(Blue Band) - Log(Green Band)')
    ax[1].set_ylabel('Depth')
    xt = (list(ax[1].get_xticks()))[:-1]

    for i,x in enumerate(xt):
        xt[i] = np.round(x,2)
    ax[1].set_title(sf.get_date().strftime("%Y%m%d%H%M%S") + ' -> tide - ' + str(sf.get_tide()) + 'm')
    xlim = (-r, r)
    ylim = ( -25,0)
    plt.setp(ax[1], xlim=xlim, ylim=ylim)


    fn = sf.get_reef_name() + '-' + sf.get_date().strftime("%Y%m%d%H%M%S")+ '.png'
    out = os.path.join(sf.get_img_path(), fn)
    plt.savefig(out)
    plt.close(fig)
    return


def corr_plot(datum,reef_name,outpath):
    r = 6
    num_blocks = int(np.ceil(np.sqrt(len(datum))))
    fig, ax = plt.subplots(num_blocks,num_blocks, figsize = (20,24))
    xlim = (-r, r)
    ylim = ( -25,0)
    plt.setp(ax, xlim=xlim, ylim=ylim)

    axlist = []
    for axl in ax:
        for axl2 in axl:
            axlist.append(axl2)


    day_keys = list(datum.keys())
    for i,dict_item in enumerate(datum.items()):
        d = dict_item[1][2]
        line = dict_item[1][0]
        meta = dict_item[1][1]

        sns.scatterplot(x = d['diff'], y = d.Height, color = 'blue', ax = axlist[i])
        sns.lineplot(x = [-r,r], y = [line(-r),line(r)], color = 'black', ax = axlist[i])
        axlist[i].set_xlabel('Log(Blue Band) - Log(Green Band)')
        axlist[i].set_ylabel('Depth')
        axlist[i].set_title(str(meta['dt'].date()))
        xt = (list(axlist[i].get_xticks()))[:-1]
        for i,x in enumerate(xt):
            xt[i] = np.round(x,2)

    fn = os.path.join(outpath,'corr_plot.png')
    plt.savefig(fn)

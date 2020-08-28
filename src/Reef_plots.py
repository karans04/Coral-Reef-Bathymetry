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
from rasterio import plot

import pandas as pd
import numpy as np
import os

def plot_median_variance_graph(medians, variances, median_threshold, variance_threshold,outpath):
    fig, ax = plt.subplots()
    sns.scatterplot(x = medians, y = variances, legend = 'full',ax = ax)
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
    reef_name = sf.get_reef_name()
    dt = sf.get_date().strftime("%Y%m%d%H%M%S")
    #plot histogram of depths
    fig, ax = plt.subplots(1,3,figsize = (28,12))
    depth_histogram(ax[0], df, reef_name)
    line_of_best_fit(ax[1],data,sf,line)
    reef_scatter(fig,ax[2],df,reef_name,data)
    fn = '{reef_name}-{date}.png'.format(reef_name = reef_name, date = dt)
    out = os.path.join(sf.get_img_path(), fn)
    plt.savefig(out)
    plt.close(fig)
    return

def aggregate_plot(data,df,sf, f):
    df['Height'] = df['median']
    reef_name = sf.get_reef_name()
    fig ,ax = plt.subplots(1,3, figsize = (28,12))
    depth_histogram(ax[0], df, reef_name, f)
    reef_scatter(fig,ax[2],df,reef_name,data,f)
    tci(sf,ax[1])
    fn = '{reef_name}-{f}.png'.format(reef_name = reef_name, f = f)
    out = os.path.join(sf.get_img_path(), fn)
    plt.savefig(out)
    plt.close(fig)

def tci(sf,ax):
    tci = sf.get_tci()
    plot.show(tci, ax=ax)



def depth_histogram(ax,df,reef_name, suffix = ''):

    df['Height'].plot.hist(bins = np.arange(-30,20,1), ax = ax)
    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('{reef_name} {suffix} Depth Histogram'.format(reef_name = reef_name, suffix = suffix))



def reef_scatter(fig,ax,df,reef_name,data, suffix = ''):
    #getting just depths between +- 45m
    df = df.loc[(df.Height <= 10) & (df.Height >= -25)]
    # df = df.loc[df.Height >= 0]
    #creating a color scale at 5m intervals
    cmap = cm.colors.ListedColormap(['black','navy','mediumblue' ,'blue','royalblue', 'dodgerblue',
                                     'skyblue','limegreen',  'lime' , 'yellow'
                                      ,'orange','tomato','red','firebrick' ,'maroon'])
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
    pts = ax.scatter(x = df.x, y = df.y, c = df.Height, s= 1, cmap = cmap, norm = norm)
    #scatter plot of the track lines from the ICESAT 2 data
    ax.scatter(x = data.x, y= data.y, s = 3, c = 'black', label = 'ICESAT-2 tracks')
    custom_lines = [Line2D([0], [0], color='black', lw=4)]
    ax.legend(custom_lines, ['ICESAT-2 tracks'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('{reef_name} {suffix} Depth Predictions (m)'.format(reef_name = reef_name, suffix = suffix))


def line_of_best_fit(ax,data,sf,line):
    r = 3
    sns.scatterplot(x = data['diff'], y = data.Height, color = 'blue', ax = ax)
    sns.lineplot(x = [-r,r], y = [line(-r),line(r)], color = 'black', ax = ax)
    ax.set_xlabel('Log(Blue Band) - Log(Green Band)')
    ax.set_ylabel('Depth')
    xt = (list(ax.get_xticks()))[:-1]

    for i,x in enumerate(xt):
        xt[i] = np.round(x,2)
    ax.set_title(sf.get_date().strftime("%Y/%m/%d %H:%M:%S") + ' -> tide - ' + str(sf.get_tide()) + 'm')
    xlim = (-r, r)
    ylim = ( -25,0)
    plt.setp(ax, xlim=xlim, ylim=ylim)

def corr_plot(datum,reef_name,outpath):
    r = 3
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:42:23 2017
 
@author: gou
"""
 
import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd 
from descartes.patch import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

     
if os.name == 'nt':
    os.chdir(r'D:\Cloud\Dropbox\PythonScripts\02_UW\50_Columbia')
elif os.name == 'posix':
    os.chdir('/Users/mgou/Dropbox/PythonScripts/02_UW/50_Columbia')
else:
    os.chdir('/home/gou/uwhydro/summaProj/summaData/scripts')
     
from crb import load_nc, month12_mean, monthly_aggregate, summa_area, \
                vic_vars, summa_vars, summa_mean, summa_define_area, summa_hruid, \
                crb_proj, crb_basemap, add_cbar
 
if os.name == 'nt':
    os.chdir('F:/UW_Postdoc/Columbia')
elif os.name == 'posix':
    os.chdir('/Volumes/T2TB/UW_Postdoc/Columbia')
else:
    os.chdir('/home/gou/uwhydro/summaProj/summaData')
      
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def aggregate_water_balance():
    nc_vic, nc_summa = load_nc()   
     
    # from day to month
    flux_name, flux_unit, state_name, state_unit = vic_vars()
    vic_ts = monthly_aggregate(nc_vic, flux_name, flux_unit, state_name, state_unit)
    vic_ts.to_netcdf('vic_monthly_ts.nc')
     
     
    flux_name, flux_unit, state_name, state_unit = summa_vars()
    summa_ts = monthly_aggregate(nc_summa, flux_name, flux_unit, state_name, state_unit)
#    summaarea = summa_area()
#    summa_ts['area'] = ('hru', summaarea)
#    mask = np.where(np.isnan(summa_ts['area']), np.nan, 1.0)
#    for f in flux_name:
#        summa_ts[f] = summa_ts[f] * mask
    summa_ts.to_netcdf('summa_monthly_ts.nc')
     
#    # examine
#    vic_ts1 = vic_mon.sel(time=1)
#    err = vic_ts1['Precipitation'] - \
#          (vic_ts1['Evaporation'] + vic_ts1['Runoff'] + vic_ts1['Baseflow']) - \
#          (vic_ts1['SWE'] + vic_ts1['SoilWat'])
 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def aggregate_12_month(tstart='1980-01-01', tend ='2010-01-01'):
#    
#    vic_daily = xr.open_dataset('vic_1980-2011_basin_daily.nc')['Soil_liquid'].resample('M', 'time', 'last').diff('time').sum('soil_layers')
#    vic_monly0 = xr.open_dataset('vic_monthly_ts.nc')['Soil_liquid'].sel(time=slice(tstart, tend)).mean(dim=['lat','lon']).sum('soil_layers')
#    vic_monly1 = xr.open_dataset('vic_monthly_ts.nc')['Soil_liquid'].sel(time=slice(tstart, tend)).sum('soil_layers', skipna=False).mean(dim=['lat','lon'])
     
    vic_ts = xr.open_dataset('vic_monthly_ts.nc')
    vic_mon = month12_mean(vic_ts.sel(time=slice(tstart, tend)))
    vic_mon['SoilWat'] = vic_mon['Soil_liquid'].sum('soil_layers', skipna=False)
    vic_mon = vic_mon.drop('Soil_liquid')
    vic_mon.to_netcdf('vic_12_mon.nc')
     
    summa_ts = xr.open_dataset('summa_monthly_ts.nc')
    summa_mon = month12_mean(summa_ts.sel(time=slice(tstart, tend)))
    summa_mon.to_netcdf('summa_12_mon.nc')
 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# function to plot multiple stacked bars
def plot_clustered_stacked(dfall, labels=None, title="",  H="/", ax=None, legend=True, linewidth=1, **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    if ax==None:
        axe = plt.subplot(111)
    else:
        axe = ax

    for df in dfall : # for each data frame
        df.plot(kind="bar",
                  #linewidth=linewidth,
                  #edgecolor='b',
                  stacked=True,
                  ax=axe,
                  legend=False,
                  #grid=False,
#                  **kwargs
               )  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    width = 1 / float(n_df + 1)  
    for i in range(0, n_df): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i*n_col:(i+1)*n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + width * i)
                rect.set_hatch(H * i) #edited part
                rect.set_width(width)
    axe.set_xticks(np.arange(0, n_ind) + (n_df - 2) * width / 2.)
    axe.set_xticklabels(df.index)
    #axe.set_title(title)
    if legend:
        plot_clustered_stacked_legend(axe, h, l, H=H, labels=labels, x=1.01, y1=0.55, y2=0.3)
    return axe, h, l

def plot_clustered_stacked_legend(axe, h, l, H="/", labels=None, x=1.01, y1=0.55, y2=0.3):
    n_col = len(set(l))
    n_df = int(len(l)/n_col)
    l1 = axe.legend(h[:n_col], l[:n_col], loc=[x, y1], frameon=False)
    
    # Add invisible data to add another legend
    n=[]
    for i in range(n_df):
        n.append(axe.bar(0.0, 0.0, width=0.0, color="gray", hatch=H * i))

    if labels is not None:
        plt.legend(n, labels, loc=[x, y2], frameon=False)
    axe.add_artist(l1)       

    
def plot_wat_bal_subbasin(tstart='1980-01-01', tend ='2010-01-01'):
#    from mpl_toolkits.axes_grid1 import ImageGrid
    # read 12 month nc
    vic_mon = xr.open_dataset('vic_12_mon.nc')    
    summa_mon = xr.open_dataset('summa_12_mon.nc') 
    
    # calculate interested hydrology variables
    vic_mon['Precip'] = vic_mon['Precipitation']
    vic_mon['Runoff'] = -vic_mon['Baseflow'] - vic_mon['Runoff'].copy() 
    vic_mon['ET'] =  -vic_mon['Evaporation']
    vic_mon['$\Delta$SoilWat'] = -vic_mon['SoilWat'] 
    vic_mon['$\Delta$SWE'] = -vic_mon['SWE']
    vic_mon = vic_mon[['Precip', 'Runoff', 'ET', '$\Delta$SoilWat', '$\Delta$SWE']]
    
    summa_mon['Precip'] = summa_mon['pptrate_mean']
    summa_mon['Runoff'] = -summa_mon['scalarAquiferBaseflow_mean'] - \
                           summa_mon['scalarSurfaceRunoff_mean'] - \
                           summa_mon['scalarSoilBaseflow_mean']
    summa_mon['ET'] =  summa_mon['scalarCanopyTranspiration_mean'] + \
                       summa_mon['scalarCanopySublimation_mean'] + \
                       summa_mon['scalarAquiferTranspire_mean'] + \
                       summa_mon['scalarCanopyEvaporation_mean'] + \
                       summa_mon['scalarSnowSublimation_mean'] + \
                       summa_mon['scalarGroundEvaporation_mean']
    summa_mon['$\Delta$SoilWat'] = -summa_mon['scalarTotalSoilIce_mean'] - \
                                    summa_mon['scalarTotalSoilLiq_mean']
    summa_mon['$\Delta$SWE'] = -summa_mon['scalarSWE_mean']
    summa_mon = summa_mon[['Precip', 'Runoff', 'ET', '$\Delta$SoilWat', '$\Delta$SWE']]
    
    # set parameters to calculate aereal SUMMA mean
    limits = -5e30, 5e30 # mm/month
    summaarea = summa_area()
    #summaarea = summa_define_area(summa_mon, summaarea, limits)
    
#    for v in summa_mon:
        
    # read tables defining relationship of subbasins and grids and hrus
    subgrid = pd.read_csv('tables/subbasin_grid.csv')
    subhru = pd.read_csv('tables/subbasin_hru.csv')
    subgrid.lat = np.round(subgrid.lat, 5)
    subgrid.lon = np.round(subgrid.lon, 5)
    
    # find lat lon indexes in vic_mon
    lat = vic_mon.lat.to_index()
    lon = vic_mon.lon.to_index()
    subgrid['ilat'] = [lat.get_loc(l) for l in subgrid.lat]
    subgrid['ilon'] = [lon.get_loc(l) for l in subgrid.lon]
    
    # find hru indexes in summa_mon
    hruid = summa_hruid().tolist()
    subhru['ihru'] = [hruid.index(h) for h in subhru.hruId]
    
    
    # start plotting
    fig, axs = plt.subplots(3, 3, sharex=True, figsize=(13.5, 9))
#    fig = plt.figure(figsize=(13.5, 9))
#    axs = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.1, cbar_mode='single', sharey=False)
    i = 0
    axs = axs.flatten()
    for (kg, gg),(kh, gh)  in zip(subgrid.groupby('Short'), subhru.groupby('Short')):
        ax = axs[i]
        print(kg, gg.shape, gh.shape)
        df_vic = pd.DataFrame({'Precip'         :  np.nanmean(vic_mon['Precip'].values[:, gg['ilat'], gg['ilon']], axis=1),
                               'Runoff'         :  np.nanmean(vic_mon['Runoff'].values[:, gg['ilat'], gg['ilon']], axis=1),                                   
                               'ET'             :  np.nanmean(vic_mon['ET'].values[:, gg['ilat'], gg['ilon']], axis=1),
                               '$\Delta$SoilWat':  np.nanmean(vic_mon['$\Delta$SoilWat'].values[:, gg['ilat'], gg['ilon']], axis=1),
                               '$\Delta$SWE'    :  np.nanmean(vic_mon['$\Delta$SWE'].values[:, gg['ilat'], gg['ilon']], axis=1)},
                               index='Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split())
        
        area_sma = summaarea[gh['ihru']]  
        df_sma = pd.DataFrame({'Precip'         :  summa_mean(summa_mon['Precip'].values[:, gh['ihru']], 
                                                              area_sma, limits, axis=1),
                               'Runoff'         :  summa_mean(summa_mon['Runoff'].values[:, gh['ihru']], 
                                                              area_sma, limits, axis=1),                                 
                               'ET'             :  summa_mean(summa_mon['ET'].values[:, gh['ihru']], 
                                                              area_sma, limits, axis=1),
                               '$\Delta$SoilWat':  summa_mean(summa_mon['$\Delta$SoilWat'].values[:, gh['ihru']], 
                                                              area_sma, limits, axis=1),
                               '$\Delta$SWE'    :  summa_mean(summa_mon['$\Delta$SWE'].values[:, gh['ihru']], 
                                                              area_sma, limits, axis=1)},
                               index='Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split())
                               
                               
        ax, h, l = plot_clustered_stacked([df_vic.loc['Oct Nov Dec Jan Feb Mar Apr May Jun Jul Aug Sep'.split(),],
                                           df_sma.loc['Oct Nov Dec Jan Feb Mar Apr May Jun Jul Aug Sep'.split(),]], 
                                           labels=['VIC','SUMMA'], ax=ax, H='///', legend=False)
#        plot_clustered_stacked([df_vic.loc['Oct Nov Dec Jan Feb Mar Apr May Jun Jul Aug Sep'.split(),],
#                                df_sma.loc['Oct Nov Dec Jan Feb Mar Apr May Jun Jul Aug Sep'.split(),],
#                                df_sma.loc['Oct Nov Dec Jan Feb Mar Apr May Jun Jul Aug Sep'.split(),]], 
#                               labels=['VIC','SUMMA','S'], H='///', legend=False)
#        ax = df_vic.loc['Oct Nov Dec Jan Feb Mar Apr May Jun Jul Aug Sep'.split(),].plot(
#            kind='bar', ax = axs[i], stacked=True, legend=False) #, edgecolor='none'
        ax.text(0.95, 0.95, kg, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        i = i + 1
    

    axs[3].set_ylabel('Monthly water balance (mm)')
    axs[7].set_xlabel('Month')
    fig.subplots_adjust(right=12/13.5)
    ax = fig.add_axes([12/13.5, 0.25, 1.5/13.5, 0.5])
    plot_clustered_stacked_legend(ax, h, l, H="///", labels=['VIC','SUMMA'], x=0.1, y1=0.55, y2=0.3)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    
    
    ax = axs[i]    
    #fig, ax = plt.subplots()
    # whole basin VIC 
    vic_mon_all = vic_mon.mean(dim=['lat','lon'])
    df_vic = pd.DataFrame({'Precip'         :  vic_mon_all['Precip'].values,
                           'Runoff'         :  vic_mon_all['Runoff'].values,                                   
                           'ET'             :  vic_mon_all['ET'].values,
                           '$\Delta$SoilWat':  vic_mon_all['$\Delta$SoilWat'].values,
                           '$\Delta$SWE'    :  vic_mon_all['$\Delta$SWE'].values},
                           index='Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split())
    
    # whole basin SUMMA 
    area_sma = summaarea 
#    limits = -5000, 5000 # mm/month
    df_sma = pd.DataFrame({'Precip'         :  summa_mean(summa_mon['Precip'].values, 
                                                          area_sma, limits, axis=1),
                           'Runoff'         :  summa_mean(summa_mon['Runoff'].values, 
                                                          area_sma, limits, axis=1),                                 
                           'ET'             :  summa_mean(summa_mon['ET'].values, 
                                                          area_sma, limits, axis=1),
                           '$\Delta$SoilWat':  summa_mean(summa_mon['$\Delta$SoilWat'].values, 
                                                          area_sma, limits, axis=1),
                           '$\Delta$SWE'    :  summa_mean(summa_mon['$\Delta$SWE'].values, 
                                                          area_sma, limits, axis=1)},
                           index='Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split())
                               
#    ax = df_vic.loc['Oct Nov Dec Jan Feb Mar Apr May Jun Jul Aug Sep'.split(),].plot(kind='bar', stacked=True, ax=axs[i], legend=False) #ax=axs[i], 
    
    
    plot_clustered_stacked([df_vic.loc['Oct Nov Dec Jan Feb Mar Apr May Jun Jul Aug Sep'.split(),],
                            df_sma.loc['Oct Nov Dec Jan Feb Mar Apr May Jun Jul Aug Sep'.split(),]], 
                            labels=['VIC','SUMMA'], ax=ax, H='///', legend=False)
    ax.text(0.95, 0.95, 'CRB', horizontalalignment='right', 
            verticalalignment='top', transform=ax.transAxes)
    
    fig.savefig('outputs/water_balance_monthly.png', dpi=600, bbox_inches='tight')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# function to plot distribution of monthly flow and storage

from collections import OrderedDict
def plot_wat_bal_map(tstart='1980-01-01', tend ='2010-01-01'):
#    from mpl_toolkits.axes_grid1 import ImageGrid
#    fig = plt.figure(figsize=(8,5))
#    axs = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1, cbar_mode='single', subplot_kw=dict(projection=columbia_crs))
#    crb_basemap(axs[0], columbia_crs, states_provinces=False, country_borders=False)
#    crb_basemap(axs[1], columbia_crs, states_provinces=False, country_borders=False)
    idx_cols = OrderedDict([('DJF', (11, 0, 1)), 
                           ('MAM', (2, 3, 4)),
                           ('JJA', (5, 6, 7)), 
                           ('SON', (8, 9, 10)),
                           ('Annual', range(12))])
    
    columbia_crs = crb_proj()
    areas = summa_area()
    
    shp_hru = gpd.read_file('mapdata/columbia_hru_output_geo.shp')
    shp_hru.to_crs(crs=columbia_crs.proj4_params, inplace=True)
    shp_hru = shp_hru[areas > 0.0]
        
    summa_mon = xr.open_dataset('summa_12_mon.nc') 
    summa_mon = summa_mon.sel(hru = areas > 0.0)
    
    summa_states = xr.open_dataset('model_results/summa_crb_all_daily.nc')[['scalarSWE_mean', 'scalarTotalSoilIce_mean', 'scalarTotalSoilLiq_mean']].sel(time=slice(tstart, tend))
    summa_states = month12_mean(summa_states)
    
    summa_mon['Precip'] = summa_mon['pptrate_mean']
    summa_mon['Runoff'] = summa_mon['scalarAquiferBaseflow_mean'] + \
                          summa_mon['scalarSurfaceRunoff_mean'] + \
                          summa_mon['scalarSoilBaseflow_mean']
    summa_mon['ET'] =  summa_mon['scalarCanopyTranspiration_mean'] + \
                       summa_mon['scalarCanopySublimation_mean'] + \
                       summa_mon['scalarAquiferTranspire_mean'] + \
                       summa_mon['scalarCanopyEvaporation_mean'] + \
                       summa_mon['scalarSnowSublimation_mean'] + \
                       summa_mon['scalarGroundEvaporation_mean']
    summa_mon['ET'] = -summa_mon['ET']
    summa_mon['SoilWat'] = summa_states['scalarTotalSoilIce_mean'] + summa_states['scalarTotalSoilLiq_mean']
    summa_mon['SWE'] = summa_states['scalarSWE_mean'] 
    
    polys = []
    icolor = []
    for geom, i in zip(shp_hru.geometry, range(shp_hru.shape[0])):
        if geom.type.startswith('Multi'):
            for poly in geom:
                polys.append(poly)
                icolor.append(i)
        else:
            polys.append(geom)
            icolor.append(i)
    
    df_rows = pd.DataFrame(dict(
        var   = ('Precip', 'ET',  'Runoff', 'SoilWat', 'SWE'),
        label = ('Precip', 'ET',  'Runoff', 'SoilWat', 'SWE'),
        cmap  = ('YlGn',   'PuRd',  'BuPu', 'Oranges', 'Blues'),
        vmin  = (0,        0,     0,        800,       0),
        vmax  = (200,      100,   100,      1200,      1000),  
        under = ('',       '',    '',       '',        'tan'),
        extd  = ('max',    'max', 'max',    'both',    'both')
    ))        
    
    pad = 40000
    xmin = shp_hru.geometry.bounds.minx.min() - pad
    ymin = shp_hru.geometry.bounds.miny.min() - pad
    xmax = shp_hru.geometry.bounds.maxx.max() + pad
    ymax = shp_hru.geometry.bounds.maxy.max() + pad
    
    
    fig, axs = plt.subplots(5, 6, figsize=(15, 18), #sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.1, wspace=0.1, width_ratios=(1,1,1,1,1,0.05)))
#    fig, axs = plt.subplots(2, 2, figsize=(5, 6), #sharex=True, sharey=True,
#                            gridspec_kw=dict(hspace=0.1, wspace=0.0, width_ratios=(1,0.05)))
    #fig.patch.set_facecolor('lightgrey')
    for i, row in df_rows.iterrows():  
        vals = summa_mon[row['var']].values
        # set color map
        norm = Normalize(row['vmin'], row['vmax'])
        cmap = plt.get_cmap(row['cmap'])
        if row['under'] != '': 
            vals = np.where(vals <= row['vmin'], vals - 0.1, vals)
            cmap.set_under(row['under'])
        
        add_cbar(cmap, norm, ax=axs[i, -1], label=row['label'], extend=row['extd'])
        for j, idx in zip(range(5), idx_cols):
            ax = axs[i, j]
            if i == 0:
                ax.set_title(idx)
            #ax.set_subplotspec(dict(projection=columbia_crs))
            #crb_basemap(ax, columbia_crs, grid=False, states_provinces=False, country_borders=False, land=False, ocean=False, lake=False)
            
            norm_val = norm(vals[idx_cols[idx],:].mean(axis=0))
            rgba = cmap(norm_val)
            
            patches = PatchCollection([PolygonPatch(p) for p in polys], linewidths=0.0, edgecolor='none', facecolors=rgba[icolor])
            ax.add_collection(patches)
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            ax.set_aspect(1.0)
            ax.patch.set_facecolor('lightgray')
            ax.patch.set_edgecolor('black')    
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print(i ,j)
            


    
    fig.savefig('outputs/month_water_balance/wal_all.png', dpi=600, bbox_inches='tight')        
        
    norm_val = norm(nc_mon['KWTroutedRunoff'][0,:].values)
    rgba = cmap(norm_val)
    #crb_basemap(ax, bsn='mapdata/ColumbiaBasin_wgs84.shp')
    crb_basemap(ax, columbia_crs, states_provinces=False, country_borders=False)
#%%
fig, ax = plt.subplots()
patches = PatchCollection([PolygonPatch(p) for p in polys], linewidths=0.0, edgecolor='none', facecolors=rgba[icolor])
ax.add_collection(patches)


#ax.set_axis_off()            
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
ax.set_xlim((xmin-30000, xmax+30000))
ax.set_ylim((ymin-30000, ymax+30000))
ax.set_aspect(1.0)
ax.patch.set_facecolor('lightgray')
ax.patch.set_edgecolor('black')

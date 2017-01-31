#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:39:05 2017

@author: gou
"""


import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

import cartopy.crs as ccrs


if os.name == 'nt':
    os.chdir(r'D:\Cloud\Dropbox\PythonScripts\02_UW\50_Columbia')
elif os.name == 'posix':
    os.chdir('/Users/mgou/Dropbox/PythonScripts/02_UW/50_Columbia')
else:
    os.chdir('/home/gou/uwhydro/summaProj/summaData/scripts')
     
from crb import crb_proj, crb_basemap, add_cbar, r2, mean_error, relative_mean_error, nash
 
if os.name == 'nt':
    os.chdir('F:/UW_Postdoc/Columbia')
elif os.name == 'posix':
    os.chdir('/Volumes/T2TB/UW_Postdoc/Columbia')
else:
    os.chdir('/home/gou/uwhydro/summaProj/summaData')    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# calculate monthly flow
reach_outlet = 17002698
nc_q = xr.open_dataset('model_results/summa_crb_streamflow_daily.nc')
nc_annual = nc_q[['KWTroutedRunoff', 'IRFroutedRunoff']].sel(time=slice('1980-01-01','2009-12-31')).resample('M', 'time', 'mean')
nc_annual.sSeg.values = nc_q.reachID.values
nc_annual.to_netcdf('model_results/summa_crb_streamflow_monthly.nc')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# calculate annual flow
reach_outlet = 17002698
nc_q = xr.open_dataset('model_results/summa_crb_streamflow_daily.nc')
np.nonzero(nc_q['reachID'].values == reach_outlet)
nc_annual = nc_q[['KWTroutedRunoff', 'IRFroutedRunoff']].sel(time=slice('1980-01-01','2009-12-31')).resample('A', 'time', 'mean')
nc_annual = nc_annual.mean('time')
nc_annual.sSeg.values = nc_q.reachID.values
nc_annual.to_netcdf('outputs/streamflow/annual_flow.nc')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# compare with observations
gages = ['Grand Coulee', 'Lower Granite', 'The Dalles']
shorts = ['GCL5N', 'LWG5N', 'TDA5N']
locations = [(-118.9816667,47.96555556), (-117.4439,46.6678), (-121.1722,45.6075)]
reaches = [17000466, 17001467, 17005424]
# check locations
columbia_crs=crb_proj()
shp_str = gpd.read_file('mapdata/stream_from_Naoki.shp')
shp_str.to_crs(crs=columbia_crs.proj4_params, inplace=True)
fig, ax = plt.subplots(1, 1, figsize=(9, 8), subplot_kw=dict(projection=columbia_crs))
#crb_basemap(ax, columbia_crs, states_provinces=False, country_borders=False, land=False)
shp_str.plot(ax=ax, color='grey', alpha=0.7)
gage_x = []
gage_y = []
for g, xy in zip(gages, locations):
    x, y = columbia_crs.transform_point(xy[0], xy[1], ccrs.PlateCarree())
    gage_x.append(x); gage_y.append(y)
    ax.text(x, y, g, zorder=99)
ax.scatter(gage_x, gage_y, c='r', alpha=1, zorder=98)

# plot obs and sim
q_obs = pd.read_csv('observation_data/naturalized_flow/NRNI_Flows_1929-2008_Corrected_08-2016.csv', header=[i for i in range(7)])
q_obs.index = pd.DatetimeIndex(q_obs.iloc[:,1])


q_sim = xr.open_dataset('model_results/summa_crb_streamflow_daily.nc')[['KWTroutedRunoff', 'IRFroutedRunoff', 'reachID']]

tstart = '1980-10-01'
tend   = '2008-09-30'

q_obs = q_obs[tstart:tend]
q_sim = q_sim.sel(time=slice(tstart, tend))

stats = []
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
for i, ob, rch, g in zip(range(3), shorts, reaches, gages):
    ax = axs[i]
    
    irch = np.nonzero(q_sim.reachID.values == rch)[0][0]
    qsim = q_sim['KWTroutedRunoff'].loc[tstart:tend, q_sim.reachID.values == rch]
    #qsim.plot(ax=ax)
    #q_sim['IRFroutedRunoff'].loc[tstart:tend, q_sim.reachID.values == rch].plot(ax=ax)
    
    qobs = (q_obs[ob] * 0.3048 ** 3)
    #qobs.plot(ax=ax, legend=False, linewidth=1.0)
    ax.set_title(g)
    if i == 1:
        ax.set_ylabel('Streamflow (m$^3$/s)')
    else:        
        ax.set_ylabel('')
        
    # calculate the stats
    qobs = qobs.values
    qsim = qsim.values
    stats.append((mean_error(qobs, qsim), relative_mean_error(qobs, qsim),
                  r2(qobs, qsim), nash(qobs, qsim)))

axs[1].legend(axs[1].lines, ['Simulated','Naturalized'])    
axs[2].set_xlabel('Time')
          
fig.savefig('outputs/streamflow/streamflow_daily.png', dpi=600, bbox_inches='tight')
pd.DataFrame(stats, index=gages, columns=['ME', 'MRE', 'R2', 'NASH'])

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
for i, ob, rch in zip(range(3), shorts, reaches):
    ax = axs[i]
    
    irch = np.nonzero(q_sim.reachID.values == rch)[0][0]
    q_sim['KWTroutedRunoff'].loc[tstart:tend, q_sim.reachID.values == rch].plot(ax=ax)
    #q_sim['IRFroutedRunoff'].loc[tstart:tend, q_sim.reachID.values == rch].plot(ax=ax)
    
    (q_obs[ob] * 0.3048 ** 3).plot(ax=ax, legend=False)
    ax.set_title(ob)
    ax.set_yscale("log", nonposy='clip')
    if i == 1:
        ax.set_ylabel('Streamflow (m$^3$/s)')
    else:        
        ax.set_ylabel('')

axs[1].legend(axs[1].lines, ['Simulated','Observed'])    
axs[2].set_xlabel('Time')
          
fig.savefig('outputs/streamflow/streamflow_daily_log.png', dpi=600, bbox_inches='tight')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot annual flow
from matplotlib.collections import LineCollection
nc_annual = xr.open_dataset(r'D:\Cloud\Dropbox\postdoc\summa\columbia\plots\streamflow\annual_flow.nc')
nc_annual['KWTroutedRunoff'].max()
nc_annual['KWTroutedRunoff'].min()
df_annual = nc_annual.to_dataframe()
df_annual.describe()

columbia_crs=crb_proj()
#shp_str = gpd.read_file('mapdata/stream_from_Naoki.shp')
shp_str = gpd.read_file(r'D:\Cloud\Dropbox\postdoc\summa\columbia\data\stream_from_Naoki.shp')
shp_str.to_crs(crs=columbia_crs.proj4_params, inplace=True)

shp_str = shp_str.set_index('seg_id2')
shp_str['KWTroutedRunoff'] = np.nan
shp_str['IRFroutedRunoff'] = np.nan
shp_str['KWTroutedRunoff'] = df_annual['KWTroutedRunoff']
shp_str['IRFroutedRunoff'] = df_annual['IRFroutedRunoff']

#shp_str_crb = shp_str[np.isnan(shp_str['KWTroutedRunoff'])>0.01]
shp_str_crb = shp_str[shp_str['KWTroutedRunoff']>0.01]


# plot annual flow
fig, ax = plt.subplots(1, 1, figsize=(9, 8), subplot_kw=dict(projection=columbia_crs))
norm = LogNorm(0.01, 1e4)
cmap = plt.get_cmap('jet')
cmap.set_bad(alpha=0.0)
norm_val = norm(shp_str_crb['KWTroutedRunoff'])
rgba = cmap(norm_val)
#crb_basemap(ax, bsn='mapdata/ColumbiaBasin_wgs84.shp')
crb_basemap(ax, columbia_crs, bsn=r'D:\Cloud\Dropbox\postdoc\summa\columbia\data\ColumbiaBasin_wgs84.shp', states_provinces=False, country_borders=False)
#shp_str[:1].plot(ax=ax, color=rgba)
str_lines = LineCollection(shp_str_crb.geometry, colors=rgba, linewidths=norm_val*2)


ax.add_collection(str_lines)

add_cbar(fig, cmap, norm, label='Mean monthly streamflow m$^3$/s', location=[0.85, 0.2, 0.025, 0.6], extend='both')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# animate monthly flow
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation

def update_month(i, nc_mon, norm, cmap, title, lines):
    norm_val = norm(nc_mon['KWTroutedRunoff'][i,:].values)
    rgba = cmap(norm_val)
    lines.set_color(rgba)
    lines.set_linewidths(norm_val*2)
    title.set_text(pd.to_datetime(nc_mon.time[i].values).strftime('%Y-%m'))
    return title, lines

nc_mon = xr.open_dataset('model_results/summa_crb_streamflow_monthly.nc')


columbia_crs=crb_proj()
#shp_str = gpd.read_file('mapdata/stream_from_Naoki.shp')
shp_str = gpd.read_file(r'D:\Cloud\Dropbox\postdoc\summa\columbia\data\stream_from_Naoki.shp')
shp_str.to_crs(crs=columbia_crs.proj4_params, inplace=True)
shp_str = shp_str.set_index('seg_id2')

# line up with the HRU Id
shp_str = shp_str.loc[nc_mon.sSeg.values,:]

# set color map
norm = LogNorm(0.01, 1e4)
cmap = plt.get_cmap('viridis_r')
cmap.set_bad(alpha=0.0)
cmap.set_under('darkgrey')


# plot annual flow
fig, ax = plt.subplots(1, 1, figsize=(9, 8), subplot_kw=dict(projection=columbia_crs))
norm_val = norm(nc_mon['KWTroutedRunoff'][0,:].values)
rgba = cmap(norm_val)
#crb_basemap(ax, bsn='mapdata/ColumbiaBasin_wgs84.shp')
crb_basemap(ax, columbia_crs, bsn=r'D:\Cloud\Dropbox\postdoc\summa\columbia\data\ColumbiaBasin_wgs84.shp', states_provinces=False, country_borders=False)
#shp_str[:1].plot(ax=ax, color=rgba)
str_lines = LineCollection(shp_str.geometry, colors=rgba, linewidths=norm_val*2)
lines = ax.add_collection(str_lines)
title = ax.set_title(pd.to_datetime(nc_mon.time[0].values).strftime('%Y-%m'), fontsize='xx-large')
add_cbar(fig, cmap, norm, label='Mean monthly streamflow m$^3$/s', location=[0.85, 0.2, 0.025, 0.6], extend='both')
ani = FuncAnimation(fig, func=lambda i: update_month(i, nc_mon, norm, cmap, title, lines), frames=range(nc_mon.time.shape[0]))
ani.save('outputs/streamflow_monthly.mp4')

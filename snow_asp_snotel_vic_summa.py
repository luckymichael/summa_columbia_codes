# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import zipfile


import cartopy.crs as ccrs
import matplotlib
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

if os.name == 'nt':
    os.chdir(r'D:\Cloud\Dropbox\PythonScripts\02_UW\50_Columbia')
    from crb import crb_proj, crb_basemap, add_cbar, load_nc, mean_error, relative_mean_error, r2, nash, summa_area
    os.chdir('F:/UW_Postdoc/Columbia')
else:
    os.chdir('/home/gou/uwhydro/summaProj/summaData/scripts')
    from crb import crb_proj, crb_basemap, add_cbar, load_nc, mean_error, relative_mean_error, r2, nash, summa_area
    os.chdir('/home/gou/uwhydro/summaProj/summaData')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# functions to read snow observation data and calculate the error stats
def read_snotel(site_dir):
    """
    read snotel data, return pandas series
    """
    df_ob = []
    for c in  filter(lambda c: c.endswith('.csv'), os.listdir(site_dir)):
        df_c = pd.read_csv(site_dir + '/' + c, skiprows=3, index_col=1, header=0)
        if 'WTEQ.I-1 (in) ' in df_c.columns:
            df_ob.append(df_c['WTEQ.I-1 (in) '])
    obs = pd.concat(df_ob)
    obs[obs == -99.9] = np.nan
    obs.index = pd.DatetimeIndex(obs.index)
    obs.index.name='Time'
    obs.name = 'Observed'
    obs = obs.resample(rule='M', how=how, axis=0)
    obs = obs * 25.4  # inch to mm
    return obs

def read_asp(asp_zip, asp_csv):
    """
    read asp zip files
    """
    with zipfile.ZipFile(asp_zip) as z:
        with z.open(asp_csv) as f:
            obs = pd.read_csv(f, skiprows=8, index_col=1, header=0)['Snow Water Equivalent']
    
    if obs.dtypes is np.dtype('O'): 
        obs[obs == ' '] = 'NaN'
        obs[obs == '******'] = 'NaN'
        
    obs = obs.astype(np.float)
    obs = obs[pd.notnull(obs.index)]
    obs.index = pd.DatetimeIndex(obs.index)
    obs.index.name='Time'
    obs.name = 'Observed'
    obs = obs.resample(rule='M', how=how, axis=0)
    return obs
#%%
how = 'first'    
def cal_snow_stat(obs, nc_summa, vic_swe, hru, lat, lon, 
                  how=how, prefix='', csv=False, plot=False, title=None):
    """
    calculate the statistics bwtween the snow observation and 
      simulation of vic and summa.
    return mean error, relative mean rrror, r2, nash of summa and vic
    """
    all_ = obs.to_frame()
    
    summa = nc_summa['scalarSWE_mean'][:,hru].load().resample('M', 'time', how)
    summa.name = 'SUMMA'    
    all_['SUMMA'] = summa.to_dataframe()['SUMMA']
        
    vic = nc_vic['SWE'].loc[:, lat, lon].resample('M', 'time', how)
    vic.name = 'VIC'
    all_['VIC'] = vic.to_dataframe()['VIC']
    
    if csv: all_.to_csv(prefix + '.csv')
    if plot: 
        ax = all_.plot()
        if title != None: ax.set_title(title)
        ax.set_ylabel('Snow Water Equivalent (mm)')
        ax.figure.savefig(prefix + '.png', dpi=600, bbox_inches='tight')
       
    
    all_.dropna(inplace=True)
    all_ = all_[all_.index.month == 4]
    obs = all_['Observed'].values
    summa = all_['SUMMA'].values
    vic = all_['VIC'].values
    
    return( mean_error(obs, summa),
            relative_mean_error(obs, summa),
            r2(obs, summa),
            nash(obs, summa),
            mean_error(obs, vic),
            relative_mean_error(obs, vic),
            r2(obs, vic),
            nash(obs, vic))
    
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# call the function to read simulation data
nc_vic, nc_summa = load_nc()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# calculate the snow stats

snow_sites = pd.read_csv('tables/snow_station_hru_grid.csv')
hru_id = pd.read_csv('tables/hruId.csv', index_col='hruId')
snow_sites['hru'] = hru_id.loc[snow_sites['hru_id2']].values

for c in ['summa_me','summa_rme','summa_r2','summa_nash', 
          'vic_me','vic_rme','vic_r2','vic_nash']:
              snow_sites[c] = np.nan


for i, r in snow_sites.iterrows():    
    #if i < 208: continue
    s = r['Station Id']
    if s[0] == 'S':
        # snotel site
        obs = read_snotel('observation_data/snow/snotel/' + s[1:])
        prefix = 'outputs/snow/monthly_snotel_' + s[1:]
        title = 'SNOTEL ' + s[1:]
    else:
        obs = read_asp(asp_zip='observation_data/snow/asp/' + s + '.zip', asp_csv=s + '.csv')
        title = 'ASP ' + s
        prefix = 'outputs/snow/monthly_asp_' + s
        
    results = cal_snow_stat(obs, nc_summa, nc_vic, 
                      hru=r['hru'], lat=r['grid_y'], lon=r['grid_x'],
                      how=how, prefix=prefix, csv=False, plot=False, title=title)
    
    snow_sites.loc[i, ['summa_me','summa_rme','summa_r2','summa_nash',
                       'vic_me','vic_rme','vic_r2','vic_nash']] = results
                       
snow_sites[['Station Id', 'hru', 'grid_x', 'grid_y',
            'summa_me','summa_rme','summa_r2','summa_nash', 
            'vic_me','vic_rme','vic_r2','vic_nash']].to_csv('outputs/snow_april_swe.csv')
            
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# prepare data to make plots
snow_stat = pd.read_csv('outputs/snow_april_swe.csv', index_col=1)
snow_sites = pd.read_csv('tables/snow_station.csv', index_col=1)
exclude_sites = '2A17P 2G03P S315 S341 S418 S446 S606 S630 S679 S711 S726 S743 S761 S791 S811 S817 S837'.split()

# remove the unwanted sites
snow_stat.drop(exclude_sites, inplace=True)
snow_stat['x'] = snow_sites['Longitude']
snow_stat['y'] = snow_sites['Latitude']
snow_stat['z'] = snow_sites['Elevation']
snow_stat.describe()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# prepare plotting

fontsize=12

def set_proj():
    return ccrs.Mercator(central_longitude=-120, min_latitude=41, max_latitude=53, globe=None)


def add_gridlines(axis,labelsize=fontsize):
    gl=axis.gridlines(draw_labels=True,
                    xlocs = [-100, -110, -115, -120, -125],
                    ylocs = [42, 44, 46, 48, 50, 52],
                    linewidth=1, color='gray', alpha=0.5, linestyle='--')

    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': labelsize}
    gl.ylabel_style = {'size': labelsize}
    return gl

# see this how to change background color: 
#    http://stackoverflow.com/questions/32200438/change-the-background-colour-of-a-projected-matplotlib-axis
def add_map_features(ax, states_provinces=True, country_borders=True, land=True, ocean=True,lake=False):
    if states_provinces==True:
        states_provinces = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_1_states_provinces_lines',
                scale='50m',
                facecolor='none')
        ax.add_feature(states_provinces, edgecolor='black', zorder = 2) #linewidth = 2

    if country_borders==True:
        country_borders = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_0_boundary_lines_land',
                scale='50m',
                facecolor='none')
        ax.add_feature(country_borders, edgecolor='black', zorder = 2, linewidth = 1)

    if land==True:
        land = cfeature.NaturalEarthFeature(
            category='physical',
            name='land',
            scale='50m',
            facecolor='gray')
        ax.add_feature(land,facecolor='lightgray', zorder = 0)

    if ocean==True:
        ocean = cfeature.NaturalEarthFeature(
            category='physical',
            name='ocean',
            scale='50m',
            facecolor='blue')
        ax.add_feature(ocean,facecolor='lightblue', zorder = 1)

    if lake==True:
        rivers_lakes = cfeature.NaturalEarthFeature(
            category='physical',
            name='rivers_lake_centerlines',
            scale='50m',
            facecolor='none')
        ax.add_feature(rivers_lakes,facecolor='lightblue', zorder = 2)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot april 1 swe error  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

models = ['summa', 'vic']
stats = ['me', 'rme', 'r2', 'nash']
stats_full = ['ME'  , 'RME', 'R$^2$'   , 'NSH']
vmins =      [-600  , -1   ,  0        ,  0]
vmaxs =      [ 600  ,  1   ,  0.91     ,  0.8]
extends =    ['both', 'max',  'neither', 'min']
# start plotting
columbia_crs=set_proj()
gdf_basin = gpd.read_file('mapdata/ColumbiaBasin_wgs84.shp')
gdf_basin.to_crs(crs=columbia_crs.proj4_params, inplace=True)

for m in models:
    for s, sf, vmin, vmax, ext in zip(stats, stats_full, vmins, vmaxs, extends):
        fname = m + '_' + s
        # get figure
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), 
                               subplot_kw=dict(projection=columbia_crs))
        ax.set_extent([-125.01, -109.5, 41, 53], ccrs.Geodetic())
        #ax.set_axis_bgcolor('lightgre..\\data\\y')
        #gl = add_gridlines(ax)
        add_map_features(ax)
        gdf_basin.plot(ax=ax, facecolor='darkgrey')
        
        sitexy = columbia_crs.transform_points(src_crs=ccrs.Geodetic(), 
                                               x=snow_stat.x.values, 
                                               y=snow_stat.y.values)
        
        cmap = plt.get_cmap('RdYlBu') #RdYlBu
        #vmin=0; vmax=1
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sc = ax.scatter(sitexy[:,0], sitexy[:,1], s=30, 
                        c=snow_stat[fname].values, vmin=vmin, vmax=vmax, 
                        cmap=cmap, alpha=1.0, zorder=99)
        
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.annotate(m.upper(), (0.03,0.95), xycoords='axes fraction')
        cbar = fig.colorbar(sc, pad=0.02, fraction=0.1, shrink=0.71, 
                            orientation='horizontal', extend=ext)
        cbar.set_label(sf)
        
        #fig.tight_layout()
        fig.savefig('outputs/snow_april_' + fname + '.png', 
                    dpi=600, bbox_inches='tight')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example showing how to turn off axis
fig, ax = plt.subplots(1, 1)
dat = np.random.randn(3,10)
sc = ax.scatter(dat[0,:], dat[1,:], s=40, 
                c=dat[2,:], 
                cmap='RdYlBu', edgecolor='none', alpha=1.0, zorder=99)
#ax.annotate( s='dat[2,:]', xy=dat[:2,:].T )
ax.set_axis_off()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
cbar = fig.colorbar(sc, pad=0.08, fraction=0.1, orientation='horizontal')
cbar.set_label('R$^2$')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# calculate monthly SWE
nc_summa = xr.open_dataset('model_results/summa_crb_all_daily.nc')
nc_swe = nc_summa['scalarSWE_mean'].sel(time=slice('1980-01-01', '2009-12-31')).resample('M', 'time', 'mean')
nc_swe.to_dataset().to_netcdf('model_results/summa_crb_swe_monthly_ts.nc')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# animate monthly flow
from descartes.patch import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

def update_month(i, nc_mon, norm, cmap, title, patches, icolor):
    norm_val = norm(nc_mon[i,:].values)
    rgba = cmap(norm_val)
    patches.set_color(rgba[icolor])
    title.set_text(pd.to_datetime(nc_mon.time[i].values).strftime('%Y-%m'))
    return title, patches

nc_mon = xr.open_dataset('model_results/summa_crb_swe_monthly_ts.nc')
shp_hru = gpd.read_file('mapdata/columbia_hru_output_geo.shp')
shp_hru.to_crs(crs=columbia_crs.proj4_params, inplace=True)

# exclude fail hru and water
areas = summa_area()
shp_hru = shp_hru[areas > 0.0]
nc_mon = nc_mon['scalarSWE_mean'][:,areas > 0.0]
nc_mon.values = np.where(nc_mon.values > 0.0, nc_mon.values, -0.1)
             
             
columbia_crs=crb_proj()

# set color map
norm = Normalize(0, 800)
cmap = plt.get_cmap('Blues')
cmap.set_bad(alpha=0.0)
cmap.set_under('tan')


norm_val = norm(nc_mon[0,:].values)
rgba = cmap(norm_val)
#crb_basemap(ax, bsn='mapdata/ColumbiaBasin_wgs84.shp')
#shp_str[:1].plot(ax=ax, color=rgba)
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
        
polys = PatchCollection([PolygonPatch(p) for p in polys], facecolors=rgba[icolor], linewidths=0.0)

# plot annual flow
fig, ax = plt.subplots(1, 1, figsize=(9, 8), subplot_kw=dict(projection=columbia_crs))
crb_basemap(ax, columbia_crs, states_provinces=False, country_borders=False)
patches = ax.add_collection(polys)
title = ax.set_title(pd.to_datetime(nc_mon.time[0].values).strftime('%Y-%m'), fontsize='xx-large')
add_cbar(fig, cmap, norm, label='Snow Water Equivalent (mm)', location=[0.85, 0.12, 0.025, 0.76], extend='both')
# animation
ani = FuncAnimation(fig, func=lambda i: update_month(i, nc_mon, norm, cmap, title, patches, icolor), frames=range(nc_mon.time.shape[0]))
ani.save('outputs/swe_monthly.mp4')

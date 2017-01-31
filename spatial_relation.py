# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 23:51:28 2016

@author: Michael Ou
"""

#result analysis
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np

#%%############################################################################
# check hru ID for each sites (ASP and SNOTEL)
os.chdir(r'D:\Cloud\Dropbox\postdoc\summa\columbia')


hru_shp = gpd.read_file('data\\columbia_hru_output_geo.shp')
hru_shp.to_crs(epsg=4326, inplace=True) # WSG 84

#Snow Water Equivalent: The water content of a snowpack at a point,
# expressed as the depth of water that would result from melting the snow.
# Normally measured in millimetres (mm).
site_snotel = pd.read_csv('ob\\list\\station_list_full.csv', index_col=0)[['Station Id' ,'Latitude', 'Longitude', 'Elevation']]
site_snotel['Station Id'] = ['S' + str(sid) for sid in site_snotel['Station Id']]
site_snotel['Elevation'] = site_snotel['Elevation'] * 0.3048

#Snow Water Equivalent: The water content of a snowpack at a point,
# expressed as the depth of water that would result from melting the snow.
# Normally measured in millimetres (mm).
site_asp = pd.read_csv('canada_snow/asp_station_list.csv')[['No.' ,'Latitude', 'Longitude', 'Elev']]
site_asp.columns = ['Station Id' ,'Latitude', 'Longitude', 'Elevation']

site_snow = pd.concat([site_snotel, site_asp], axis=0)
site_snow.to_csv('snow_station.csv')

point_snow = gpd.GeoDataFrame(geometry=[Point(r.Longitude, r.Latitude) for i, r in site_snow.iterrows()])

snow_hru = gpd.sjoin(hru_shp, point_snow, how="inner", op='intersects')
snow_hru['Elev_diff'] = np.abs(snow_hru['demmean'].values - site_snow['Elevation'].values[snow_hru['index_right'].values])
snow_hru.sort_values('Elev_diff', inplace=True)

site_hru = snow_hru.groupby('hru_id2').first()
site_hru['Elev_diff'] = site_hru['demmean'].values - site_snow['Elevation'].values[site_hru['index_right'].values]
site_hru['Station Id'] = site_snow['Station Id'].values[site_hru['index_right'].values]
site_hru.drop('geometry', inplace=True, axis=1)
site_hru.to_csv('snow_station_hru.csv')

###############################################################################
# check grid id for each site (ASP and SNOTEL)
s = site_snow.set_index('Station Id')
point_snow = gpd.GeoDataFrame(geometry=[Point(s.loc[i, 'Longitude'],s.loc[i, 'Latitude'])  for i in site_hru['Station Id']])
shp_grid = gpd.read_file(r'..\summaData\blivneh\blivneh_grid_crb.shp')
shp_grid.to_crs(epsg=4326, inplace=True) # WSG 84
snow_grid = gpd.sjoin(point_snow, shp_grid, how="inner", op='intersects')
snow_grid = snow_grid[~snow_grid.index.duplicated(keep='first')]
snow_grid.sort_index(inplace=True)
site_hru['grid_x'] = ((snow_grid['xmin'] + snow_grid['xmax']) * 0.5).values
site_hru['grid_y'] = ((snow_grid['ymin'] + snow_grid['ymax']) * 0.5).values

site_hru.to_csv('snow_station_hru_grid.csv')

#%%############################################################################
# check ameriflux
os.chdir(r'D:\Cloud\Dropbox\postdoc\summa\columbia\data')

shp_bsn = gpd.read_file('basin_selected.shp')
shp_hru = gpd.read_file('columbia_hru_output_geo.shp')
shp_grid = gpd.read_file('../../summaData/blivneh/blivneh_grid_crb.shp')
shp_bsn.to_crs(epsg=4326, inplace=True) # WSG 84
shp_hru.to_crs(epsg=4326, inplace=True) # WSG 84
shp_grid.to_crs(epsg=4326, inplace=True) # WSG 84

site = pd.read_csv('../ameriflux/sites.csv')
points = gpd.GeoDataFrame(geometry=[Point(r.LOCATION_LONG, r.LOCATION_LAT) for i, r in site.iterrows()])

# spatial inference
site_hru = gpd.sjoin(points, shp_hru, how="inner", op='intersects')
site_hru = site_hru[~site_hru.index.duplicated()]
site_grid = gpd.sjoin(points, shp_grid, how="inner", op='intersects')
site_grid = site_grid[~site_grid.index.duplicated()]

aflux_hru_grid = site.loc[site_hru.index,:]
aflux_hru_grid['hruId'] = site_hru['hru_id2']
aflux_hru_grid['grid_lat'] = site_grid.loc[:, ['ymin', 'ymax']].mean(axis=1)
aflux_hru_grid['grid_lon'] = site_grid.loc[:, ['xmin', 'xmax']].mean(axis=1)
aflux_hru_grid.to_csv('../ameriflux/aflux_hru_grid.csv')
# to shapefile
aflux_hru_grid['geometry'] = points.loc[site_hru.index]
gpd.GeoDataFrame(aflux_hru_grid).to_file('ameri_flux.shp')
#%%############################################################################
# check selected basin
os.chdir(r'D:\Cloud\Dropbox\postdoc\summa\columbia\data')

shp_bsn = gpd.read_file('basin_selected.shp')
shp_hru = gpd.read_file('columbia_hru_output_geo.shp')
shp_grid = gpd.read_file(r'..\..\summaData\blivneh\blivneh_grid_crb.shp')

shp_bsn.to_crs(epsg=4326, inplace=True) # WSG 84
shp_hru.to_crs(epsg=4326, inplace=True) # WSG 84
shp_grid.to_crs(epsg=4326, inplace=True) # WSG 84


#shp_hru_projected = shp_hru.to_crs(epsg=26911) # NAD83 UTM11N
#shp_hru.to_file('columbia_hru_output_geo.shp')

###############################################################################
###############################################################################
###############################################################################
grid_point = gpd.GeoDataFrame(geometry=shp_grid.centroid, crs=shp_grid.crs)
grid_intersect = gpd.tools.sjoin(grid_point, shp_bsn, how='inner')
grid_intersect['lon'] = [p.x for p in grid_intersect['geometry']]
grid_intersect['lat'] = [p.y for p in grid_intersect['geometry']]
grid_intersect[['lat','lon','Short']].to_csv('..\\subbasin_grid.csv')
#fig, ax = plt.subplots(figsize=(10,12))
#shp_gpoint.plot(ax=ax)
#shp_bsn[0:1].plot(ax=ax, facecolor='r', zorder=100)



###############################################################################
###############################################################################
###############################################################################
hru_point = gpd.GeoDataFrame(geometry=shp_hru.centroid, crs=shp_hru.crs)
hru_intersect = gpd.tools.sjoin(hru_point, shp_bsn, how='inner')
hru_intersect['hruId'] = shp_hru['hru_id2'][hru_intersect.index.values]
hru_intersect[['hruId', 'Short']].to_csv('subbasin_hru.csv')
hru_intersect['yrPPTmm'] = hru_intersect['yrPPTmm'] * hru_intersect['areaKm2']
hru_intersect['yrTEMPc'] = hru_intersect['yrTEMPc'] * hru_intersect['areaKm2']
###############################################################################
###############################################################################
###############################################################################
bsn_final = shp_bsn.set_index(keys='Short')
bsn_final['Cell Count'] = grid_intersect.groupby('Short').count()['geometry']
bsn_final['HRU Count'] = hru_intersect.groupby('Short').count()['geometry']
bsn_final['Elevation'] = hru_intersect.groupby('Short').mean()['demmean']
bsn_final['Lat'] = [p.y for p in shp_bsn.centroid]
bsn_final['Lon'] = [p.x for p in shp_bsn.centroid]
bsn_final['HRUarea'] = hru_intersect.groupby('Short').sum()['areaKm2']
bsn_final['Anual Tempeature C'] = hru_intersect.groupby('Short').sum()['yrTEMPc'] / bsn_final['HRUarea']
bsn_final['Anual Precip mm'] = hru_intersect.groupby('Short').sum()['yrPPTmm'] / bsn_final['HRUarea']
bsn_final[['Name',
           'Lat',
           'AreaSqKm',
           'Elevation',
           'Cell Count',
           'HRU Count',
           'Anual Precip mm',
           'Anual Tempeature C']].to_csv('basin_selected.csv')

#%%############################################################################
###############################################################################
###############################################################################

import os
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


if os.name == 'nt':
    os.chdir(r'D:\Cloud\Dropbox\PythonScripts\02_UW\50_Columbia')
    from crb import crb_proj, crb_basemap, add_cbar
    os.chdir('F:/UW_Postdoc/Columbia')
else:
    os.chdir('/home/gou/uwhydro/summaProj/summaData/scripts')
    from crb import crb_proj, crb_basemap, add_cbar
    os.chdir('/home/gou/uwhydro/summaProj/summaData')
    
os.chdir(r'D:\Cloud\Dropbox\postdoc\summa\columbia\data')

crb_crs = crb_proj()

shp_bsn = gpd.read_file('basin_selected.shp')
shp_bsn.to_crs(crs=crb_crs.proj4_params, inplace=True)

# prepare snow site data
snow_sites = pd.read_csv(r'D:\Cloud\Dropbox\postdoc\summa\columbia\snow_station.csv', index_col=1)
snow_stat = pd.read_csv(r'D:\Cloud\Dropbox\postdoc\summa\columbia\plots\snow\map\snow_april_swe.csv', index_col=1)
#snow_sites.set_index('Station Id')
exclude_sites = '2A17P 2G03P S315 S341 S418 S446 S606 S630 S679 S711 S726 S743 S761 S791 S811 S817 S837'.split()
snow_stat.drop(exclude_sites, inplace=True)
snow_stat['x'] = snow_sites['Longitude']
snow_stat['y'] = snow_sites['Latitude']
snow_stat.describe()

sitexy = crb_crs.transform_points(src_crs=ccrs.Geodetic(), x=snow_stat.x.values, y=snow_stat.y.values)
snow_stat['x'] = sitexy[:,0]
snow_stat['y'] = sitexy[:,1]
snotel = snow_stat.loc[ [sid.startswith('S') for sid in snow_stat.index.values],:] 
asp    = snow_stat.loc[[not sid.startswith('S') for sid in snow_stat.index.values],:]

# prepare for ameriflux site data
sites = pd.read_csv(r'D:\Cloud\Dropbox\postdoc\summa\columbia\ameriflux\aflux_hru_grid.csv', index_col=1)
included = 'US-Me2 US-Me4 US-Me5 US-Me6 US-MRf US-Wrc'.split()
sites = sites.loc[included,:]        
sitexy = crb_crs.transform_points(src_crs=ccrs.Geodetic(), x=sites.LOCATION_LONG.values, y=sites.LOCATION_LAT.values)
sites['x'] = sitexy[:,0]
sites['y'] = sitexy[:,1]


fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(projection=crb_crs))
crb_basemap(ax, crb_crs, bsn=r'D:\Cloud\Dropbox\postdoc\summa\columbia\data\ColumbiaBasin_wgs84.shp', 
            states_provinces=False, country_borders=False)

# plot select basin
shp_bsn.plot(ax=ax)

for g, s in zip(shp_bsn.geometry, shp_bsn['Short']):
    #if s in ['UCO', 'KTL', 'UMA'] :
    if s not in ['BAU', 'SNK']:
        ax.text(g.bounds[2] + 10000, (0.7*g.bounds[1] + 0.3*g.bounds[3]), s)
    else:
        #ax.text(0.5*(g.bounds[0] + g.bounds[2]), (0.5 * g.bounds[3] + 0.5 * g.bounds[3]) + 0.5, s, horizontalalignment='center', verticalalignment='bottom')
        ax.text(0.5*(g.bounds[0] + g.bounds[0])-10000, (0.5 * g.bounds[1] + 0.5 * g.bounds[3]) , s, horizontalalignment='right', verticalalignment='bottom')
    
sc_asp    = ax.scatter(asp['x'].values,    asp['y'].values,    s=25, c='g', marker='P')
sc_snotel = ax.scatter(snotel['x'].values, snotel['y'].values, s=25, c='b', marker='X')
sc_aflux  = ax.scatter(sites['x'].values,  sites['y'].values,  s=25, c='r', marker='D')

ax.legend([sc_asp, sc_snotel, sc_aflux], ['ASP', 'SNOTEL', 'AmeriFlux'], loc='upper right')

# ameriflux site

fig.savefig(r'D:\Cloud\Dropbox\postdoc\summa\columbia\plots\basin_selected.png', dpi=600, bbox_inches='tight')

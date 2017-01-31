#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:28:05 2017

@author: gou

summer useful functions
"""
import xarray as xr
import numpy as np

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import geopandas as gpd
from matplotlib.colorbar import ColorbarBase
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# function to load data
def load_nc(ncvic='/raid3/oriana/bpa/runs/historical/vic/160912_september2016_calibration/nc/calibrated_all_fluxes.19500101-20111231.nc',
            ncsumma='/home/gou/uwhydro/summaProj/summaData/summa_crb_all_daily.nc'):
    nc_vic = xr.open_dataset(ncvic)
    nc_summa = xr.open_dataset(ncsumma)
    return nc_vic, nc_summa

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# parameters;
i_water = 17

def summa_area(arr_file='inputs/summa_zLocalAttributes_columbia_gru.nc', nan_value=0.0):
    ihru_failed = np.array([8523, 8670, 8689, 8737, 9039, 9862, 11130, 11132, 11204, 11484])
    nc_arr = xr.open_dataset(arr_file)
    area = nc_arr['HRUarea'].values
    veg_type = nc_arr['vegTypeIndex'].values
    area[ihru_failed - 1] = nan_value
    area[veg_type == i_water] = nan_value
    return area


def summa_hruid(arr_file='inputs/summa_zLocalAttributes_columbia_gru.nc'):
    nc_arr = xr.open_dataset(arr_file)
    return nc_arr['hruId'].values

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# all the model units are converted to mm/day or mm
def summa_vars():
    flux_name = ['scalarCanopySublimation_mean', 'scalarCanopyEvaporation_mean', 'scalarCanopyTranspiration_mean',
                 'scalarSnowSublimation_mean', 'scalarGroundEvaporation_mean', 'scalarAquiferTranspire_mean',
                 'scalarSurfaceRunoff_mean', 'scalarAquiferBaseflow_mean', 'scalarSoilBaseflow_mean',
                 'scalarRainPlusMelt_mean', 'pptrate_mean']
    flux_unit = [86400, 86400, 86400,
                 86400, 86400, 86400000,
                 86400000, 86400000, 86400000,
                 86400000, 86400]

    state_name = ['scalarAquiferStorage_mean', 'scalarTotalSoilLiq_mean', 'scalarTotalSoilIce_mean',
                  'scalarSWE_mean']
    state_unit = [1000, 1.0, 1.0,
                  1.0]
    return(flux_name, flux_unit, state_name, state_unit)

def vic_vars():
    flux_name = ['Precipitation', 'Evaporation', 'Runoff', 'Baseflow']
    flux_unit = [1.0, 1.0, 1.0, 1.0]

    state_name = ['Soil_liquid', 'SWE']
    state_unit = [1.0, 1.0]
    return(flux_name, flux_unit, state_name, state_unit)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# function to calculate monthly (12 month) mean and monthly difference
def month12_mean(nc):
    nc.time.values = nc['time.month'].values
    ncf = []
    for v in nc:
        if 'time' in nc[v].dims and len(nc[v].dims) > 1:
            ncm = xr.concat([nc[v].sel(time=i).mean('time') for i in range(1,13)], dim='time')
            ncf.append(ncm)
        elif v not in nc.dims:
            ncf.append(nc[v])
    ncf = xr.merge(ncf)
    ncf.time.values = range(1, 13)
    return(ncf)


def monthly_aggregate(nc, flux_name, flux_unit, state_name, state_unit):
    ts = []
    for f, u in zip(flux_name, flux_unit):
        ts.append((nc[f].resample('M', 'time', 'sum', skipna=False)[1:]) * u)

    for s, u in zip(state_name, state_unit):
        ts.append(nc[s].resample('M', 'time', 'last').diff('time') * u)

    nc_ts = xr.merge(ts)
    return(nc_ts)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# calculate the areal mean of HRUs
def summa_mean(var, areas, limits, axis):
    """
    calculate the areal mean
    avoid the values beyond the limits
    """
    i = 0
    for a in areas:
        if any(var[:,i] < limits[0]) or any(var[:,i] > limits[1]):
            areas[i] = 0.0
        i += 1
    return(np.nansum(var * areas, axis=axis) / np.nansum(areas))

def summa_define_area(nc, areas, limits, nan_value=0.0):
    """
    set area to nan_value for HRUs having values beyond the limits
    """
    ih = []
    for v in nc:
        if v in nc.dims: continue
        ith = np.argwhere( (nc[v].values < limits[0]) | (nc[v].values > limits[1]) )
        #print(v, ith.shape)
        ih.extend(ith[:,1].tolist())
    for i in set(ih): areas[i] = nan_value
    return(areas)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# equations to calculate the stat

def mean_error(obs, sim):
    omean = np.array(obs).mean()
    smean = np.array(sim).mean()
    return smean - omean

def relative_mean_error(obs, sim):
    omean = np.array(obs).mean()
    smean = np.array(sim).mean()
    return (smean - omean) / omean

def r2(obs, sim):
    obs = np.array(obs)
    sim = np.array(sim)
    n = sim.shape[0]
    omean = obs.mean()
    smean = sim.mean()
    up = ((obs * sim).sum() - n * omean * smean ) ** 2
    low = ((obs ** 2).sum() - n * omean **2) * ((sim ** 2).sum() - n * smean ** 2)
    return up / low

def nash(obs, sim):
    obs = np.array(obs)
    sim = np.array(sim)

    omean = obs.mean()
    up = ((sim - obs) ** 2).sum()
    low = ((omean - obs) ** 2).sum()
    return 1.0 - up / low

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot basemap of CRB

def crb_proj():
    return ccrs.Mercator(central_longitude=-120, min_latitude=41, max_latitude=53, globe=None)

def crb_basemap(ax, crs, bsn=None, bsn_bg='darkgrey', grid=True, extent=[-125.01, -109.5, 41, 53],
                states_provinces=True, country_borders=True, land=True, ocean=True, lake=False):

    # add map features
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

    if grid:
        # add grid lines
        gl=ax.gridlines(draw_labels=True,
            xlocs = [-100, -110, -115, -120, -125],
            ylocs = [40, 42, 44, 46, 48, 50, 52, 54],
            linewidth=1, color='gray', alpha=0.5, linestyle='--')

        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #gl.xlabel_style = {'size': labelsize}
        #gl.ylabel_style = {'size': labelsize}

    # set axis extent
    ax.set_extent(extent, ccrs.Geodetic())

    # add basin shpe file if desired
    if bsn != None:
        shp_bsn = gpd.read_file(bsn)
        shp_bsn.to_crs(crs=crs.proj4_params, inplace=True)
        shp_bsn.plot(ax=ax, facecolor=bsn_bg)

    return ax
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# add color bar to the plot
def add_cbar(cmap, norm, label='', ax=None, fig=None, fontsize=11, location=[0.85, 0.1, 0.025, 0.8], **kwargs):
    if ax == None:
        ax = fig.add_axes(location)
    ax.tick_params(labelsize=fontsize)
    cbar_abs = ColorbarBase(ax, cmap=cmap, norm=norm, **kwargs).set_label(label=label, size=fontsize+2)
    return cbar_abs

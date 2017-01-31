#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:26:37 2017

@author: gou
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile


if os.name == 'nt':
    os.chdir('F:/UW_Postdoc/Columbia')
else:    
    os.chdir('/home/gou/uwhydro/summaProj/summaData/scripts')
    from crb import load_nc, mean_error, relative_mean_error, r2, nash
    os.chdir('/home/gou/uwhydro/summaProj/summaData')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# parameters
how = 'mean' # how to calculate the monthly fluxes


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# functions to read ameriflux observation data
def parse_time(ts):
    """
    %timeit parse_time(199401010000)
    100000 loops, best of 3: 7.17 µs per loop 
    """
    year = ts//100000000
    month = ts//1000000 - year * 100
    day = ts//10000 - year * 10000 - month * 100
    hour = ts//100 - year * 1000000 - month * 10000 - day * 100
    minute = ts - year * 100000000 - month * 1000000 - day * 10000 - hour * 100
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)    


def parse_time0(ts):
    """
    %timeit parse_time0(199401010000)
    1000 loops, best of 3: 200 µs per loop
    """
    return pd.to_datetime(str(ts), format='%Y%m%d%M%S')
    
def read_aflux(aflux_zip, starttime=None, endtime=None):
    """
    read ameriflux zip files
    """
    with zipfile.ZipFile(aflux_zip) as z:
        for zf in z.namelist():
            if zf.endswith('.csv'):
                with z.open(zf) as f:
                    obs = pd.read_csv(f, header=0, comment='#')
            else:
                with z.open(zf) as f:
                    info = pd.read_excel(f)
    
    lat  = info['DATAVALUE'][[('LAT'  in v) for v in info['VARIABLE']]]
    lon  = info['DATAVALUE'][[('LONG' in v) for v in info['VARIABLE']]]
    elev = info['DATAVALUE'][[('ELEV' in v) for v in info['VARIABLE']]]
    obs1 = obs.replace({'H': {-9999: np.nan},
                 'LE':{-9999: np.nan},
                 'G': {-9999: np.nan},
                 'NETRAD': {-9999: np.nan}})[['H','LE','G', 'NETRAD']]
    obs1.index = pd.DatetimeIndex(obs.iloc[:,0].apply(parse_time))
    
    obs1.dropna(how='all', inplace=True)
    
    obs1.index.name = 'time'
    obs1.name = 'observed'
    
    return obs1, lat, lon, elev
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# calculate the error stats    
def cal_aflux_stat(obs, nc, obs_vars, sim_vars, ylabs=None, prefix='', csv=False, 
                   plot=False, title=None, starttime=None, endtime=None):
    """
    calculate the statistics bwtween the flux observation and 
      simulation of summa.
    return mean error, relative mean error, r2, nash of summa
    """
    
    if starttime != None and endtime == None: 
        obs1 = obs[starttime:]
        nc1 = nc[sim_vars].sel(time=slice(starttime, None))
    if starttime != None and endtime != None: 
        obs1 = obs[starttime:endtime]
        nc1 = nc[sim_vars].sel(time=slice(starttime, endtime))
    if starttime == None and endtime != None: 
        obs1 = obs[:endtime]
        nc1 = nc[sim_vars].sel(time=slice(None, endtime))
        
    nc1['scalarSenHeatTotal'] = -nc1['scalarSenHeatTotal']
    nc1['scalarLatHeatTotal'] = -nc1['scalarLatHeatTotal']
    
    
    #nc1 = nc1.resample('H', 'time', 'mean')
    #obs1 = obs1.resample(rule='H', how='mean', axis=0)
    if ylabs == None:
        ylabs = [None] * len(obs_vars)
    for ov, sv, yl in zip(obs_vars, sim_vars, ylabs):
        if plot:
            fig, ax = plt.subplots()
            #ax.plot(obs1[ov], label='Observed')
            #ax.plot(nc1.time, -nc1[sv], label='Simulated')
            nc1[sv].plot(ax=ax)
            obs1[ov].plot(ax=ax)
            ax.legend(ax.lines, ['Simulated', 'Observed'])
            ax.set_title(title)
            if yl != None:
                ax.set_ylabel(yl)
            fig.savefig(prefix + '_' + ov + '.png', dpi=600, bbox_inches='tight')
        #ax.set_xlim(starttime, endtime)
        
        for e in ['me','rme','r2','nash']:
            sites[v + '_' + e] = np.nan
     
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# calculate the snow stats
obs_vars = ['H', 'LE', 'G', 'NETRAD']
sim_vars = ['scalarSenHeatTotal', 'scalarLatHeatTotal', 'scalarGroundAbsorbedSolar', 'scalarGroundNetNrgFlux']
obs_vars = ['H', 'LE']
sim_vars = ['scalarSenHeatTotal', 'scalarLatHeatTotal']
ylabs = ['Sensible heat flux (W m$^{-2}$)','Latent heat flux (W m$^{-2}$)']

sites = pd.read_csv('tables/aflux_hru_grid.csv')

for v in obs_vars:
    for e in ['me','rme','r2','nash']:
        sites[v + '_' + e] = np.nan

ziplist = os.listdir('observation_data/ameriflux')
for i, r in sites.iterrows():    
    #if i < 208: continue
    site = r['SITE_ID']
    idx = np.nonzero([site in z for z in ziplist])[0]
    nc_file = 'model_results/ameri_flux/aflux_' + site + '.nc'
    if idx > 0 and os.path.exists(nc_file):
        # read data
        obs, lat, lon, elev = read_aflux(aflux_zip='observation_data/ameriflux/' + ziplist[idx[0]])
        # check if there is a useful weekly data
        for yy in range(r['TOWER_BEGAN'], 2012):
            t1 = pd.date_range(str(yy) + '-05-01', str(yy) + '-10-25')
            t2 = pd.date_range(str(yy) + '-05-07', str(yy) + '-10-31')
            for tt1, tt2 in zip(t1, t2):
                found = np.count_nonzero(np.isnan(obs.loc[tt1:tt2, obs_vars])) < 10 and obs.loc[tt1:tt2].shape[0] > 200
                if found: break
            if found: break
        # if no useful obs, skip this site    
        if not found: 
            print(site + ' has no useful data')
            continue
        print(site, ziplist[idx[0]])
        nc = xr.open_dataset(nc_file).squeeze('hru')
        prefix = 'outputs/ameriflux/' + site
        title = site
        cal_aflux_stat(obs, nc, obs_vars, sim_vars, ylabs, prefix=prefix, csv=False, 
                   plot=True, title=title, starttime=tt1, endtime=tt2)
    else:
        print(site + ' data not found')
        
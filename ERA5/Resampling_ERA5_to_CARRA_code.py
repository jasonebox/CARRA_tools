# -*- coding: utf-8 -*-
"""
@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)
Few modifications needed to be run with other datasets
-> modified by Armin Dachauer and Jason Box
"""
import numpy as np
import pandas as pd
import os
from pyproj import Proj, transform
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
# import cfgrib
from datetime import datetime


if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/CARRA/CARRA_ERA5_events'
    heavy_data_path='/Users/jason/0_dat/ERA5/events/'

os.chdir(base_path)

# %% CARRA coordinates

def lon360_to_lon180(lon360):

    # reduce the angle  
    lon180 =  lon360 % 360 
    
    # force it to be the positive remainder, so that 0 <= angle < 360  
    lon180 = (lon180 + 360) % 360;  
    
    # force into the minimum absolute value residue class, so that -180 < angle <= 180  
    lon180[lon180 > 180] -= 360
    
    return lon180

# CARRA West grid dims
ni = 1269 ; nj = 1069

# read lat lon arrays
fn = './meta/CARRA/2.5km_CARRA_west_lat_1269x1069.npy'
lat = np.fromfile(fn, dtype=np.float32)
clat_mat = lat.reshape(ni, nj)

fn = './meta/CARRA/2.5km_CARRA_west_lon_1269x1069.npy'
lon = np.fromfile(fn, dtype=np.float32)
lon_pn = lon360_to_lon180(lon)
clon_mat = lon_pn.reshape(ni, nj) 

# fn='./meta/CARRA/CARRA_W_domain_ice_mask.nc'
# ds=xr.open_dataset(fn)
# mask = np.array(ds.z)

# %% reproject 4326 (lat/lon) CARRA coordinates to 3413 (orth polar projection in meters)

#from lat/lon to meters
inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:3413')

x1, y1 = clon_mat.flatten(), clat_mat.flatten()
cx, cy = transform(inProj, outProj, x1, y1)
cx_mat = cx.reshape(ni, nj) 
cy_mat = cy.reshape(ni, nj)

cols, rows = np.meshgrid(np.arange(np.shape(clat_mat)[1]), 
                         np.arange(np.shape(clat_mat)[0]))

CARRA_positions = pd.DataFrame({'rowc': rows.ravel(),
                                'colc': cols.ravel(),
                                'xc': cx_mat.ravel(),
                                'yc': cy_mat.ravel()})

# ,
# 'maskc': mask.flatten()}
#import CARRA datset
# ds = xr.open_dataset(raw_path+'tp_2012.nc')
# CARRA_data = np.array(ds.tp[0, :, :]).flatten()

# %% load ERA5 coordinates 

#ERA 5 elevation data
fn='./meta/ERA5/ERA5_mask_and_terrain.nc'
ds=xr.open_dataset(fn)

#from lat/lon to meters
inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:3413')

niE=np.shape(ds.latitude.values)[0]
njE=np.shape(ds.longitude.values)[0]

lat_mesh, lon_mesh = np.meshgrid(ds.latitude.values, ds.longitude.values)
x1, y1 = lon360_to_lon180(lon_mesh.flatten()), lat_mesh.flatten()
ex, ey = transform(inProj, outProj, x1, y1)
ex_mat = ex.reshape(niE, njE) 
ey_mat = ey.reshape(niE, njE)

cols2, rows2 = np.meshgrid(np.arange(np.shape(ds.latitude.values)[0]), np.arange(np.shape(ds.longitude.values)[0]))  
lat_e, lon_e = np.meshgrid(ds.latitude.values, ds.longitude.values)
ERA5_positions = pd.DataFrame({
"row_e": rows2.ravel(),
"col_e": cols2.ravel(),
"lon_e": lon_e.ravel(),
"lat_e": lat_e.ravel(),})

#load ERA5 dataset
fn='./data_raw/ERA5/tcwv/202206_3hourly_tcwv.grib'
fn='/Users/jason/0_dat/ERA5/events/tpsft2m/20210814-17_3hourly_tpsft2m.grib'
fn='/Users/jason/0_dat/ERA5/events/tpsf/20210814-17_3hourly_tpsf.grib'
# fn='/Users/jason/0_dat/ERA5/events/tzuv/20210814-17_3hourly_tzuv.grib'
ds=xr.open_dataset(fn,engine='cfgrib')
# ds = cfgrib.open_dataset(fn)

print(ds.variables)
print(np.shape(ds.tp))
#%%

steps=['-21','-00','-03','-06']
steps=['00','01','02','03']
times=ds.variables['time']
dates=[]
# date_strings=[]
# str_dates = [i.strftime("%Y-%m-%dT%H:%M") for i in time]
for i,time in enumerate(times):
    for step in range(4):
        # print(str(np.array(time)))
        timex=pd.to_datetime(times[i].to_numpy())
        
        print(timex.strftime('%Y-%m-%d-%H'),step)
        #%%
        # dates.append(str(np.array(time)))
        # temp=str(np.array(time))[0:13]

        # date_strings.append(temp)
        # dates.append(datetime.strptime(temp, "%Y-%m-%dT%H"))
        # print(date_time_obj)
    
        # print(x.strftime("%Y-%m-%dT%H:%M"))
        # print(date_strings)
        
        # dtime=pd.to_datetime(date_strings,format="%Y-%m-%dT%H")
        
        choice='tp'
            
        if choice=='t2m':
            ERA_data = np.array(ds.t2m[i, :, :])-273.15
        if choice=='tp':
            ERA_data = np.array(ds.tp[i,step, :, :])*1000
        
        # print(np.shape(ERA_data))
        
        
        #  nearest neighbours
        
        #dataset to be upscaled -> ERA5
        nA = np.column_stack((ex_mat.ravel(), ey_mat.ravel()) ) 
        #dataset to provide the desired grid -> CARRA
        nB = np.column_stack((cx_mat.ravel(), cy_mat.ravel()))
        
        btree = cKDTree(nA)  #train dataset
        dist, idx = btree.query(nB, k=1)  #apply on grid
        
        #collocate matching cells
        CARRA_cells_for_ERA5 = ERA5_positions.iloc[idx]
        
        # Output resampling key ERA5 data in CARRA grid
        path_tools='/tmp/'
        CARRA_cells_for_ERA5.to_pickle(path_tools+'resampling_key_ERA5_to_CARRA.pkl')
        
        outpath='/Users/jason/0_dat/ERA5/events/resampled/'+choice+'/'
        os.system('mkdir -p '+outpath)
    
        print(date_strings[i]+steps[step])
        #  visualisation
        
        new_grid= ERA_data[CARRA_cells_for_ERA5.col_e, CARRA_cells_for_ERA5.row_e]
        new_grid = np.rot90(new_grid.reshape(ni, nj).T)
        plt.close()
        plt.imshow(new_grid)
        plt.title(date_strings[i]+steps[step])
        plt.colorbar()
        DPI=100
        ly='p'
                
        if ly == 'p':
            figpath=outpath
            # figpath='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/Figs/ERA5/'
            # os.system('mkdir -p '+figpath)
            plt.savefig(figpath+date_strings[i]+steps[step]+'.png', bbox_inches='tight', pad_inches=0.04, dpi=DPI, facecolor='w', edgecolor='k')
            # plt.savefig(figpath+select_period+'JJA_'+hgt+'z_anom.eps', bbox_inches='tight')
        
        # new_grid.astype(dtype=np.float16).tofile(outpath+'/'+date_strings[i]+steps[step]+'_1269x1069.npy',)


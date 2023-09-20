# -*- coding: utf-8 -*-
"""

followed by:
    
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
from datetime import timedelta


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

if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/'
    heavy_data_path='/Users/jason/0_dat/ERA5/events/'

os.chdir(base_path)


fn='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/stats/rf_extremes.csv'
os.system('ls -lF '+fn)
os.system('head '+fn)
os.system('tail '+fn)
df=pd.read_csv(fn)

print(df.columns)
#%%
# minval=300 ;scaling_factor=3000
# v=np.where(df.maxlocalrate>minval)

v=np.where(df.Gt_overall>2.2)
# v=np.where(df.Gt_overall>2.7)
# v=v[0]
# print(v[0])

# print()
minval=np.min(df.maxlocalrate[v[0]]) ; scaling_factor=1500
listx=[v[0][3]] # 2021
# listx=[v[0][-4]] # 2017
# listx=v[0]
# listx=[v[0][0]] # 2012a
listx=[v[0][12]] # 2017

model='CARRA'
model='ERA5'

if model=='CARRA':
    choices=['tpt2m']
else:
    choices=['tpsf']

for ii,i in enumerate(listx):
    print(model,df.date[i],df.Gt_overall[i],ii)
    #%%
    ni=1269 ; nj=1069
    
    # fn='./meta/CARRA/2.5km_CARRA_west_elev_1269x1069.npy'
    # elevx=np.fromfile(fn, dtype=np.float32)
    # elevx=elevx.reshape(ni, nj)
    # elevx=np.rot90(elevx.T)
    
    # global plot settings
    th=1
    font_size=18
    plt.rcParams['axes.facecolor'] = 'k'
    plt.rcParams['axes.edgecolor'] = 'k'
    plt.rcParams["font.size"] = font_size
    plt.rcParams['axes.facecolor']='w'
    plt.rcParams['savefig.facecolor']='w'
    
    
    timex=pd.to_datetime(df.date[i]) 

    casex=timex.strftime('%Y-%m-%d')
    casex2=timex.strftime('%d %B, %Y')
    
    timex=pd.to_datetime(df.date[i]) 

    ddx=int(timex.strftime('%j'))
    ddx0=ddx-1 ; ddx1=ddx+1
    ymd=timex.strftime('%Y-%m-%d')
    year=timex.strftime('%Y')
    month=timex.strftime('%m')
    day=timex.strftime('%d')

    timex0=pd.to_datetime(df.date[i])- timedelta(days=1)
    ymd0=timex0.strftime('%Y-%m-%d')
    day_before=timex0.strftime('%d')
    
    timex1=pd.to_datetime(df.date[i])+ timedelta(days=1)
    ymd1=timex1.strftime('%Y-%m-%d')
    
    print(i,ymd0,ymd,ymd1,ddx,df.lon[i],df.lat[i],df.Gt_overall[i],df.maxlocalrate[i])

    
    day_before=str(int(day)-1).zfill(2)
    day_three=str(int(day)+1).zfill(2)
    # day_eight=str(int(day)+6).zfill(2)
    
    choice='tpsf'
    opath='/Users/jason/0_dat/ERA5/events/tpsf/'
    ofile=opath+str(year)+str(month).zfill(2)+str(day_before).zfill(2)+'-'+day_three+'_3hourly_'+choice+'.grib'
    print(ofile)
    #%%
    # casex='2017-09-14'; casex2='14 September, 2017'
    # casex='2022-06-17'; casex2='17 June, 2022'
    #load ERA5 dataset
    
    # fn='/Users/jason/0_dat/ERA5/events/tpsf/20210814-17_3hourly_tpsf.grib'
    # fn='/Users/jason/0_dat/ERA5/events/tzuv/20210814-17_3hourly_tzuv.grib'
    ds=xr.open_dataset(ofile,engine='cfgrib')
    # ds = cfgrib.open_dataset(fn)
    
    # print(ds.variables)
    # print(np.shape(ds.tp))
    #%%
    
    steps=['-21','-00','-03','-06']
    steps=['00','01','02','03']
    times=ds.variables['time']
    # for time in times:
    #     # print(str(np.array(time)))
    #     dates.append(str(np.array(time)))
    #     temp=str(np.array(time))[0:16]
    #     date_strings.append(temp)
    #     dates.append(datetime.strptime(temp, "%Y-%m-%dT%H:%M"))
    # print(dates)

    # str_dates = [i.strftime("%Y-%m-%dT%H:%M") for i in time]
    cc=0
    sumx=np.zeros((1269, 1069))
    for i,time in enumerate(times):
        for step in range(4):
            if ((cc>0) & (cc<25)):
                temp=str(np.array(time))[0:16]
                x=datetime.strptime(temp, "%Y-%m-%dT%H:%M")
                timex0=pd.to_datetime(times[i].to_numpy()) + timedelta(hours=step*3-6)
                if x.strftime('%H')=='18':
                    timex0=pd.to_datetime(times[i].to_numpy()) + timedelta(hours=(step*3)-6)
                timex0=timex0 + timedelta(hours=9)
    
                date_stringx=timex0.strftime('%Y-%m-%d-%H')
                print(cc,date_stringx)
                
                var_choice='tp'
                var_choice='rf'
                    
                if var_choice=='t2m':
                    ERA_data = np.array(ds.t2m[i, :, :])-273.15
                if var_choice=='tp':
                    ERA_data = np.array(ds.tp[i,step, :, :])*1000*3
                if var_choice=='rf':
                    ERA_data = np.array(ds.tp[i,step, :, :])*1000*3-np.array(ds.sf[i,step, :, :])*1000*3  
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
                
                outpath='/Users/jason/0_dat/ERA5/events/resampled/'+var_choice+'/'
                os.system('mkdir -p '+outpath)
            
                #  visualisation
                
                new_grid= ERA_data[CARRA_cells_for_ERA5.col_e, CARRA_cells_for_ERA5.row_e]
                new_grid = np.rot90(new_grid.reshape(ni, nj).T)
                plt.close()
                plt.imshow(new_grid,vmin=0,vmax=12)
                plt.axis('off')
                plt.title(var_choice+' '+date_stringx)
                plt.colorbar()
                DPI=200
                ly='p'
                if ly=='x':
                    plt.show()                    
                if ly == 'p':
                    figpath=outpath
                    # figpath='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/Figs/ERA5/'
                    # os.system('mkdir -p '+figpath)
                    plt.savefig(figpath+date_stringx+'.png', bbox_inches='tight', pad_inches=0.04, dpi=DPI, facecolor='w', edgecolor='k')
                
                new_grid.astype(dtype=np.float16).tofile(outpath+'/'+date_stringx+'_1269x1069.npy',)
                sumx+=new_grid
                if ((cc==8)or(cc==16)or(cc==24)):
                    sumx.astype(dtype=np.float16).tofile(outpath+'/'+timex0.strftime('%Y-%m-%d')+'_1269x1069.npy',)
                    
                    plt.close()
                    plt.imshow(sumx,vmin=0,vmax=48)
                    plt.axis('off')
                    plt.title(var_choice+' '+timex0.strftime('%Y-%m-%d'))
                    plt.colorbar()
                    DPI=200
                    ly='p'
                    if ly=='x':
                        plt.show()                    
                    if ly == 'p':
                        figpath=outpath
                        # figpath='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/Figs/ERA5/'
                        # os.system('mkdir -p '+figpath)
                        plt.savefig(figpath+timex0.strftime('%Y-%m-%d')+'.png', bbox_inches='tight', pad_inches=0.04, dpi=DPI, facecolor='w', edgecolor='k')
                    sumx=np.zeros((1269, 1069))
       
            cc+=1
    

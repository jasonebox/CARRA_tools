#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 08:08:56 2020

@author: Jason Box, GEUS, jeb@geus.dk

ERA5 vis for a pressure level with temperature 

"""
# import netCDF4
# from netCDF4 import Dataset,num2date
# import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
#import basemap: see https://stackoverflow.com/questions/52295117/basemap-import-error-in-pycharm-keyerror-proj-lib
from mpl_toolkits.basemap import Basemap
import cfgrib
from datetime import datetime
import pandas as pd
from PIL import Image

# if os.getlogin() == 'jason':
#     base_path = '/Users/jason/Dropbox/SICE_showcase_study/ERA5/'
#     heavy_data_path='/Users/jason/0_dat/ERA5/raw/'
#     # heavy_data_path='/Users/jason/Dropbox/SICE_showcase_study/ERA5/raw/'

# os.chdir(base_path)

nj=1440 ; ni=721

th=1
font_size=9
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size

fn='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/data_raw/ERA5/tzuv/202206_3hourly_tzuv.grib'
fn='/Users/jason/0_dat/ERA5/events/tzuv/2022061319_3hourly_tzuv.grib'
fn='/Users/jason/0_dat/ERA5/events/tzuv/20210814-17_3hourly_tzuv.grib'
ds = cfgrib.open_dataset(fn)
# nc = xr.open_dataset(fn,engine='cfgrib')
lat = ds.variables['latitude'][:]
lon = ds.variables['longitude'][:]
times=ds.variables['time']

# fn='/Users/jason/0_dat/ERA5/raw/event_1h/ERA5_202108_3hourly_850hPa_t.grib'
dst = cfgrib.open_dataset(fn)
print(dst)
#%%
dates=[]
date_strings=[]
# str_dates = [i.strftime("%Y-%m-%dT%H:%M") for i in time]
for time in times:
    # print(str(np.array(time)))
    dates.append(str(np.array(time)))
    temp=str(np.array(time))[0:13]
    print(temp)
    date_strings.append(temp)
    dates.append(datetime.strptime(temp, "%Y-%m-%dT%H"))
    # print(date_time_obj)

    # print(x.strftime("%Y-%m-%dT%H:%M"))
print(date_strings)

dtime=pd.to_datetime(date_strings,format="%Y-%m-%dT%H")
print(print)

#%%
lons,lats= np.meshgrid(lon,lat) # for this dataset, longitude is 0 through 360, so you need to subtract 180 to properly display on map
                
# proj_choice='stereo'
# if proj_choice=='stereo':
#    m = Basemap(projection='npstere',boundinglat=60,lon_0=0,resolution='l')

m = Basemap(width=4000000,height=4600000,
        resolution='l',projection='stere',\
        lat_0=63,lon_0=-50.)

x,y = m(lons,lats)
xx, yy = m.makegrid(nj, ni, returnxy=True)[2:4]

#%%
i=0

for i in range(len(date_strings)):
    if i>=0:
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(figsize=(5, 5))

        U=np.array(ds.variables['u'][i,:,:])#.reshape(ni,nj)
        V=np.array(ds.variables['v'][i,:,:])#.reshape(ni,nj)
        
        print(date_strings[i])
        
        c0=0.4
        m.drawcoastlines(linewidth=0.6,color=[c0,c0,c0])
        m.drawparallels(np.arange(-90.,120.,30.), linewidth=0.5)
        m.drawmeridians(np.arange(0.,420.,50.), linewidth=0.5)
        
        cmapx='bwr'
        mcmap = plt.get_cmap(cmapx)
        # mcmap.set_over('m')
        
        # dx=10
        # lo=-150 ; hi=abs(lo)+dx
        # clevs=np.arange(lo,hi,dx)
        # anom_z[anom_z<lo]=lo
        # anom_z[anom_z>hi]=hi
        # mine=m.contourf(x,y,anom_z,clevs,cmap=mcmap)
        
        dx=1
        lo=-16 ; hi=abs(lo)+dx
        clevs=np.arange(lo,hi,dx)
        # inv=np.where(anom>hi);anom[inv]=hi
        # anom[anom<lo]=lo
        # anom[anom>hi]=hi
        mine=m.contourf(x,y,np.array(dst.variables['t'][i,:,:])-273.15,levels=clevs,cmap=mcmap,extend='both')
        
        skip_interval=5
        skip=(slice(None,None,skip_interval),slice(None,None,skip_interval))
        
        m.quiver(x[skip],y[skip], U[skip], V[skip], scale_units='inches', scale=200, color='k',width=0.003,zorder=1)
        # https://stackoverflow.com/questions/33637693/how-to-use-streamplot-function-when-1d-data-of-x-coordinate-y-coordinate-x-vel/33640165#33640165
        # m.streamplot(xx,yy, U, V, color=speed, cmap=plt.cm.autumn, linewidth=0.5*speed)
        # m.streamplot(mapx, mapy, ugrid, vgrid, color='r', latlon=True)
        
        plt.clim(lo,hi)
        # ax.set_title(msg)
        
        do_colorbar=1
        
        if do_colorbar:
            clb = plt.colorbar(fraction=0.035, pad=0.05)
            clb.ax.set_title('°C',fontsize=font_size,c='k')
            clb.ax.tick_params(labelsize=font_size,labelrotation = 'auto')
    
        
        # anom_t,mean_u_for_target_year=anoms(ds,'t',ni,nj,select_period)
        # dx=1
        # lo=-12 ; hi=abs(lo)+dx
        # clevs=np.arange(lo,hi,dx)
        # anom_t[anom_t<lo]=lo
        # anom_t[anom_t>hi]=hi
        # temp=m.contourf(x,y,anom_t,clevs,cmap='bwr',linewidths=th/2.)
        
        cc=0
        xx0=0.03 ; yy0=0.04
        mult=1.2
        color_code='k'
        props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')
        plt.text(xx0, yy0, dtime[i].strftime("%Y-%m-%d %H"),
                fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5

        cc=0
        xx0=1.015 ; yy0=0.02 ; dy2=-0.04
        mult=0.85
        color_code='grey'
        plt.text(xx0, yy0+cc*dy2, 'ERA5 850 hPa vectors @climate_ice',
                fontsize=font_size*mult,color=color_code,rotation=90,transform=ax.transAxes) ; cc+=1.5

        
        DPI=200
        ly='p'
                
        if ly == 'p':
            figpath='/Users/jason/0_dat/ERA5/events/Figs/tzuv/'
            # figpath='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/Figs/ERA5/'
            os.system('mkdir -p '+figpath)
            plt.savefig(figpath+date_strings[i]+'.png', bbox_inches='tight', pad_inches=0.04, dpi=DPI, facecolor='w', edgecolor='k')
            # plt.savefig(figpath+select_period+'JJA_'+hgt+'z_anom.eps', bbox_inches='tight')


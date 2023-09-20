#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 08:08:56 2020

@author: Jason Box, GEUS, jeb@geus.dk

Output monthly ERA5 temperatures for regional polygons drawn unambiguously in photoshop with ocean buffer but obtaining land using ERA5 mask data

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

if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/CARRA/CARRA_ERA5_events'
    heavy_data_path='/Users/jason/0_dat/ERA5/events/'

os.chdir(base_path)

th=1
font_size=9
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size



fn=heavy_data_path+'/tzuv/2022061319_3hourly_tzuv.grib'
fn='/Users/jason/0_dat/ERA5/events/tcwv/20220923-26_3hourly_tcwv.grib'
fn='/Users/jason/0_dat/ERA5/events/rf/2022092231_3hourly_tzuv.grib'
ds = cfgrib.open_dataset(fn)
# nc = xr.open_dataset(fn,engine='cfgrib')
lat = ds.variables['latitude'][:]
lon = ds.variables['longitude'][:]
times=ds.variables['time']

#%%
# fn=heavy_data_path+'tcwv/2022061319_3hourly_tcwv.grib'
fn='/Users/jason/0_dat/ERA5/events/tcwv/20220923-26_3hourly_tcwv.grib'
ds_wv = cfgrib.open_dataset(fn)

# dates = netCDF4.num2date(time[:], time.)
# datesx = netCDF4.num2date(time[:], time.units)

dates=[]
date_strings=[]
# str_dates = [i.strftime("%Y-%m-%dT%H:%M") for i in time]
for time in times:
    # print(str(np.array(time)))
    dates.append(str(np.array(time)))
    temp=str(np.array(time))[0:16]
    date_strings.append(temp)
    dates.append(datetime.strptime(temp, "%Y-%m-%dT%H:%M"))
    # print(date_time_obj)

    # print(x.strftime("%Y-%m-%dT%H:%M"))
print(date_strings)

lons,lats= np.meshgrid(lon,lat) # for this dataset, longitude is 0 through 360, so you need to subtract 180 to properly display on map
                
proj_choice='stereo'
# proj_choice='ortho'
# m = Basemap(projection='npstere',boundinglat=50,lon_0=0,resolution='l')
if proj_choice=='ortho':
    m = Basemap(projection='ortho',lon_0=-44,lat_0=45,resolution='l')
    skip_interval=10
if proj_choice=='stereo':
    m = Basemap(width=5000000,height=5900000,
            resolution='l',projection='stere',\
            lat_0=55,lon_0=-44.)
    skip_interval=5

x,y = m(lons,lats)
xx, yy = m.makegrid(1440, 721, returnxy=True)[2:4]


ihour=0 ; fhour=5*24
ihour=0 ; fhour=len(times)

n_hours=fhour-ihour+1

#%%
cc=0
for i in range(1):
# for i in range(len(dates)):
# for i in range(ihour,fhour):
# for i in range(fhour-50,fhour):
# for i in range(ihour,ihour+1):
    date = datetime.strptime(str(date_strings[i][0:16]), "%Y-%m-%dT%H:%M")
    msg=str(date.year)+"-"+str(date.month).zfill(2)+"-"+str(date.day).zfill(2)+"-"+str(date.hour).zfill(2)

    if msg!='x2021-08-13-12':
        plt.close()
        plt.clf()
        fig, ax = plt.subplots()#figsize=(8, 14))
    
     
        # z = ds.variables['z'][i,:,:]/9.80665
    
        lo=5400; hi=6000
        prefix="ERA5 850 hPa geopotential height, "
        prefix=""
    
        # plotvar[plotvar>=30]=30
        # print(dates[i])
    

        tcwv = ds_wv.variables['tcwv'][i,:,:]
        var='tcwv'
        lo=0; hi=40
        prefix="total column water vapor, "
        dx=(hi-lo)/20
        units='mm'
        namx='tcwv'
            
        U = np.array(ds.variables['u'][i,:,:])
        V = np.array(ds.variables['v'][i,:,:])
        # plotvar = np.array(z)
        plotvar = np.array(tcwv)
        
        clevs = np.arange(lo,hi,dx)
        # inv=np.where(lats<50)
        # plotvar[inv[0]]=np.nan
        # print("min",np.nanmin(plotvar),"max",np.nanmax(plotvar))
        plotvar[plotvar>=np.max(clevs)]=np.max(clevs)-0.01
        plotvar[plotvar<=lo]=lo
        
        # plt.imshow(plotvar)
        # plt.colorbar()
        m.drawcoastlines(linewidth=0.3,color='k')
        m.drawparallels(np.arange(-90.,120.,30.), linewidth=0.3)
        m.drawmeridians(np.arange(0.,420.,50.), linewidth=0.3)
    
        cmapx='cividis'
        # cmapx='viridis'
        # cmapx='plasma'
        # cmapx='magma'
        mcmap = plt.get_cmap(cmapx)
        # mcmap.set_over('m')
        
        mine=m.contourf(x,y,plotvar,clevs,cmap=mcmap,linewidths=th/2.)
        
        skip=(slice(None,None,skip_interval),slice(None,None,skip_interval))
        du_vect=1
        if du_vect:
            # m.quiver(x[skip],y[skip], U[skip], V[skip], scale_units='xy', scale=1.,width=2, color='r',zorder=10)
            m.quiver(x[skip],y[skip], U[skip], V[skip], scale_units='inches', scale=190, color='w',width=0.004,zorder=1)
    # https://stackoverflow.com/questions/33637693/how-to-use-streamplot-function-when-1d-data-of-x-coordinate-y-coordinate-x-vel/33640165#33640165
        # m.streamplot(xx,yy, U, V, color=speed, cmap=plt.cm.autumn, linewidth=0.5*speed)
        # m.streamplot(mapx, mapy, ugrid, vgrid, color='r', latlon=True)
    
        plt.clim(lo,hi)
    
        do_savefig_anim=0
        do_colorbar=1
        if do_savefig_anim:do_colorbar=1
        
        if do_colorbar:
            clb = plt.colorbar(fraction=0.046/2., pad=0.04)
            clb.ax.set_title(units,fontsize=font_size,c='k')
            clb.ax.tick_params(labelsize=font_size,labelrotation = 'auto')
        
        do_annotate=1
        
        if do_annotate:
            cc=0
            xx0=1.015 ; yy0=0.02 ; dy2=-0.04
            mult=0.9
            color_code='grey'
            plt.text(xx0, yy0+cc*dy2, 'ERA5 850 hPa vectors',
                    fontsize=font_size*mult,color=color_code,rotation=90,transform=ax.transAxes) ; cc+=1.5
    
        cc=0
        xx0=0.64 ; yy0=0.04
        mult=0.9
        color_code='k'
        props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')
        plt.text(xx0, yy0, msg,
                fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5
        
        xx0=0.05 ; yy0=0.94
        mult=0.9
        color_code='k'
        
    
        # lyx='x'
        # if msg=='2021-08-13-12':
        #     plt.text(xx0, yy0, '(a)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5        
        #     lyx='p'
        # if msg=='2021-08-14-06':
        #     plt.text(xx0, yy0, '(b)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5        
        #     lyx='p'
    
        # if msg=='2021-08-16-06':
        #     plt.text(xx0, yy0, '(c)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5  
        #     lyx='p'
    
        # if msg=='2021-08-17-12':
        #     plt.text(xx0, yy0, '(d)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5  
        #     lyx='p'
    
        # if msg=='2021-08-18-06':
        #     plt.text(xx0, yy0, '(e)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5  
        #     lyx='p'

        # if msg=='2021-08-19-12':
        #     plt.text(xx0, yy0, '(f)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5  
        #     lyx='p'
    
        # if msg=='2021-08-20-06':
        #     plt.text(xx0, yy0, '(g)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5  
        #     lyx='p'
    
        # if msg=='2021-08-21-12':
        #     plt.text(xx0, yy0, '(h)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5  
        #     lyx='p'
    
        # if msg=='2021-08-27-18':
        #     plt.text(xx0, yy0, '(i)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5  
        #     lyx='p'

        # if msg=='2021-08-30-18':
        #     plt.text(xx0, yy0, '(j)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5  
        #     lyx='p'     
     
        # if msg=='2021-08-16-06':
        #     plt.text(xx0, yy0, '(c)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5  
    
        # if msg=='2021-08-16-06':
        #     plt.text(xx0, yy0, '(c)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5  
    
        # if msg=='2021-08-16-06':
        #     plt.text(xx0, yy0, '(c)',
        #             fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5  
    
    
        # plt.text(xx0, yy0+cc*dy2, 'ERA5 data, 850 hPa vectors, @climate_ice',
        #         fontsize=font_size*mult,color=color_code,transform=ax.transAxes) ; cc+=1.5
        
        # xx0=0.5 ; yy0=0.5 ; dy=0.1 ; dx=0.2
        # currentAxis = plt.gca()
        # currentAxis.add_patch(Rectangle((0. , .0), cc/n_hours, 0.01,alpha=1,transform=ax.transAxes))
        cc+=1
        
        lyx='p'
        ly=lyx

        DPI=200
        if do_savefig_anim:
            ly='p'
            # DPI=150

        if ly == 'x':
            plt.show() 
    

        if ly == 'p':
            figpath='/Users/jason/0_dat/ERA5/events/Figs/tcwv/'
            os.system('mkdir -p '+figpath)
            event=date.strftime('%Y-%m-%d-%H')
            # if do_savefig_anim:
            plt.savefig(figpath+event+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
            # else:
            #     # figpath='./ERA5/Figs/'
            #     # os.system('mkdir -p '+figpath)
            #     plt.savefig('/tmp/t.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
                
                # # ----------  Crop Image
     
                # im1 = Image.open('/tmp/t.png', 'r')
                # width, height = im1.size
                # # Setting the points for cropped image
                # left = 25
                # top = 14
                # right = width-14 #1.4 #4.3
                # bottom = height-25
                
                # # Cropped image of above dimension
                # im1 = im1.crop((left, top, right, bottom))
                # back_im = im1.copy()
                
                # back_im.save(figpath+namx+'_'+event+'.png')  
        
    #%%
make_gif=0
nam='_ERA5_6hourly_10-31_Aug_2021_'+namx+'_'+str(DPI)+'DPI'
if make_gif:
    print("making gif")
    animpath=figpath        
    inpath=figpath
    msg='convert  -delay 40  -loop 0   '+inpath+'2021*.png  '+animpath+proj_choice+''+nam+'.gif'
    os.system(msg)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads 3h t2m and accumulated precipitation from CARRA grib file obtained from CDS using:

/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/src/gather_event_3h_from_extremes_list.py

computes 3h rainfall, daily total rainfall for ERA5 and CARRA, obtains daily total and outputs to .npy files

upstream for ERA5 is /Users/jason/Dropbox/CARRA/CARRA_ERA5_events/src/ERA5/Resampling_ERA5_to_CARRA_code.py

updated Nov 2022
@author: Jason Box, GEUS, jeb@geus.dk
"""
import numpy as np
import os
from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import calendar
from datetime import timedelta

ch=2 # tp
# ch=1 # rf
# ch=0 # t2m
prt_time=0
tst_plot=0

years=np.arange(2020,2021).astype('str')
# years=np.arange(1998,1999).astype('str')
# years=np.arange(2021,2022).astype('str')
# years=np.arange(2017,2018).astype('str')

# for a later version that maps the result
ni=1269 ; nj=1069

path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
raw_path='/Users/jason/0_dat/CARRA_raw/'
raw_path='/Volumes/LaCie/0_dat/CARRA/CARRA_raw/'
outpath='/Users/jason/0_dat/CARRA/output/annual/' 
outpath='/Volumes/LaCie/0_dat/CARRA/output/annual/' 

os.chdir(path)


def get_rf(t2m,tp):
    # rain phasing
    rainos=0.
    x0=0.5 ; x1=2.5
    # x0=-2 ; x1=0
    x0-=rainos
    x1-=rainos
    y0=0 ; y1=1
    a1=(y1-y0)/(x1-x0)
    a0=y0-a1*x0

    f=np.zeros((ni,nj))
    # np.squeeze(t2m, axis=0)
    v=np.where(((t2m>x0)&(t2m<x1)))
    f[v]=t2m[v]*a1+a0
    v=np.where(t2m>x1) ; f[v]=1
    v=np.where(t2m<x0) ; f[v]=0
    
    rf=tp*f
    return rf

# plot function
def plt_x(model,var,lo,hi,nam,timestamp_string,units):
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.imshow(var,vmin=lo,vmax=hi)
    plt.title(timestamp_string)
    plt.axis('off')
    clb = plt.colorbar(fraction=0.046/2., pad=0.08)
    clb.ax.set_title(units,fontsize=7)
    ly='p'
    fig_path='/Users/jason/0_dat/CARRA_temp/'
    if ly=='p':
        plt.savefig(fig_path+'_'+nam+'_'+timestamp_string+'_'+model+'.png', bbox_inches='tight', dpi=200)#, facecolor=bg, edgecolor=fg)
    else:
        plt.show()


fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)


fn='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/stats/rf_extremes.csv'
df=pd.read_csv(fn)

print(df.columns)

# minval=300 ;scaling_factor=3000
# v=np.where(df.maxlocalrate>minval)

v=np.where(df.Gt_overall>2.2)
v=np.where(df.Gt_overall>2.7)
# v=v[0]
# print(v[0])

# print()
minval=np.min(df.maxlocalrate[v[0]]) ; scaling_factor=1500
# listx=[v[0][-1]] # 2021
# listx=[v[0][-4]] # 2017
# listx=v[0]
listx=[v[0][0]] # 2012a

model='CARRA'

if model=='CARRA':
    choices=['tpt2m']
else:
    choices=['tpsf','tzuv','tcwv']

for i in listx:
    print(df.date[i])
    timex=pd.to_datetime(df.date[i]) 
    # timex=pd.to_datetime('17/06/2022') # force this date
    # timex=[]

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

    for choice_index,choice in enumerate(choices):
    
        # path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
        
        # os.chdir(path)
        # ofile="./ancil/events/"+varnams[j]+"_events.csv"
        # # read in events df
        # df=pd.read_csv(ofile)
        # print(df.columns)
        
        # path='./Figs/event/'+varnams[j]+'/'
        
        
        # files = sorted(glob(path+"*.png"), reverse=True)
        
        # numpy.savetxt("foo.csv", a, delimiter=",")
        
        # os.system('mkdir -p /Users/jason/0_dat/ERA5/events/')
        # os.system('mkdir -p /Users/jason/0_dat/ERA5/events//')
        
        # print(files)
            
        yearx=[]
        monx=[]
        dayx=[]
        Gt=[]
    
        day_before=str(int(day)-1).zfill(2)
        day_three=str(int(day)+1).zfill(2)
        # day_eight=str(int(day)+6).zfill(2)
        print(choice,year,month,day_before,day,day_three)


        opath='/Users/jason/0_dat/'+model+'/events/'+choice+'/'
        fn=opath+str(year)+str(month).zfill(2)+str(day_before).zfill(2)+'-'+day_three+'_3hourly_'+choice+'.grib'
        # fn='/Users/jason/0_dat/CARRA/events/tpt2m/20220616-18_3hourly_tpt2m.grib'
        print(fn)
        ds=xr.open_dataset(fn,engine='cfgrib')
        t2m_all=ds.variables['t2m'].values#-273.15
        tp_all=ds.variables['tp'].values
        times=ds.variables['time'].values
        
        print(ds.variables)

        # print(np.shape(t2m))
        
        
        # for i,time in enumerate(times):
        #     if i!=1002:
        #%%
        cc=0
        for i in range(3):
            sum_CARRA=np.zeros((ni,nj))
            sum_ERA5=np.zeros((ni,nj))
            for j in range(8):
                print(i,j,cc)
            # if i!=1002:
                timex=pd.to_datetime(times[cc])
                # print()
                do_ERA5=0
                if do_ERA5:
                    fn='/Users/jason/0_dat/ERA5/events/resampled/rf/'+timex.strftime('%Y-%m-%d-%H')+'_1269x1069.npy'
                    os.system('ls -lF '+fn)
                    rfERA5=np.fromfile(fn, dtype=np.float16)
                    rfERA5=rfERA5.reshape(ni, nj)
                # var=np.rot90(tp_all[cc,:,:].T)
                rf=get_rf(t2m_all[cc,:,:]-273.15,tp_all[cc,:,:])
                timestamp=timex.strftime('%Y-%m-%d-%H')
                plt_x('CARRA',np.rot90(rf.T),0,20,'rf',timestamp,'mm')

                sum_CARRA+=rf
                if do_ERA5:
                    sum_ERA5+=rfERA5
                cc+=1
            timestamp=timex.strftime('%Y-%m-%d')
            plt_x('CARRA',np.rot90(sum_CARRA.T),0,20,'rf',timestamp,'mm')
            if do_ERA5:
                plt_x('ERA5',sum_ERA5,0,20,'rf',timestamp,'mm')
            if i==1:
                CARRA_or_ERA5='CARRA'
                outpath='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/data_raw/'+CARRA_or_ERA5+'/event/'
                sum_CARRA.astype('float16').tofile(outpath+'rf_'+timestamp+'_1269x1069.npy')
                if do_ERA5:
                    CARRA_or_ERA5='ERA5'
                    outpath='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/data_raw/'+CARRA_or_ERA5+'/event/'
                    sum_ERA5.astype('float16').tofile(outpath+'rf_'+timestamp+'_1269x1069.npy')
                # print(i,timex)
                
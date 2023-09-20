#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:31:10 2021

@author: jeb
"""
import cdsapi
import os
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import calendar
from datetime import timedelta

if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/'

os.chdir(base_path)

choices=['tpsf','tzuv','tcwv']

# choice_index=1
# choice=choices[choice_index]

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
listx=v[0]
for i in listx:
    print(df.date[i])
    # %%
    # co_index=(df.maxlocalrate[i]-200)/(np.max(df.maxlocalrate)-200)*255
    timex=pd.to_datetime(df.date[i])
    # timex=pd.to_datetime('17/06/2022') # kludge to force this date
    # print(df.date[i])
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
        
        opath='/Users/jason/0_dat/ERA5/events/'+choice+'/'
        # opath='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/data_raw/ERA5/'+choice+'/'
        os.system('mkdir -p '+opath)
        
        ofile=opath+'/'+str(year)+str(month).zfill(2)+str(day_before).zfill(2)+'-'+day_three+'_3hourly_'+choice+'.grib'
    
        c = cdsapi.Client()
    
        if choice=='tcwv':
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'grib',
                    'variable': 'total_column_water_vapour',
                    'year': year,
                    'month': str(month).zfill(2),
                    'day': [
                        str(int(day_before)).zfill(2),
                        str(day).zfill(2),
                        str(int(day_three)).zfill(2),
                        ],
                    'time': [
                        '00:00', '03:00', '06:00',
                        '09:00', '12:00', '15:00',
                        '18:00', '21:00',
                        ],
                },
                ofile)
    
        if choice=='tzuv':        
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'grib',
                    'variable': [
                        'temperature', 'u_component_of_wind', 'v_component_of_wind',
                    ],
                    'pressure_level': '850',
                    'year': year,
                    'month': str(month).zfill(2),
                    'day': [
                        str(int(day_before)).zfill(2),
                        str(day).zfill(2),
                        str(int(day_three)).zfill(2),
                        ],
                    'time': [
                        '00:00', '03:00', '06:00',
                        '09:00', '12:00', '15:00',
                        '18:00', '21:00',
                        ],
                },
                ofile)
    
        if choice=='tpsf':        
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'grib',
                        'variable': ['snowfall', 'total_precipitation',],
                        'year': year,
                        'month': str(month).zfill(2),
                        'day': [
                            str(int(day_before)).zfill(2),
                            str(day).zfill(2),
                            str(int(day_three)).zfill(2),
                            ],
                        'time': [
                            '00:00', '03:00', '06:00',
                            '09:00', '12:00', '15:00',
                            '18:00', '21:00',
                        ],
                    },
                ofile)

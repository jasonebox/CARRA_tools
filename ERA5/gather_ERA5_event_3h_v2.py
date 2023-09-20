#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:31:10 2021

@author: jeb
"""
import cdsapi
import os
import pandas as pd

if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/'

os.chdir(base_path)

choices=['tcwv','z']
choices=['tzuv','tcwv']
choices=['tpsft2m']
choices=['tpsf']

choice_index=0
choice=choices[choice_index]

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
    
    for choice in choices:
        year='2022' ; month='6' ; day='17'
        year='2021' ; month='8' ; day='14'
        
        day_before=str(int(day)-1).zfill(2)
        day_three=str(int(day)+1).zfill(2)
        day_four=str(int(day)+2).zfill(2)
        day_five=str(int(day)+3).zfill(2)
        # day_six=str(int(day)+4).zfill(2)
        # day_seven=str(int(day)+5).zfill(2)
        last_day=day_five
        # day_eight=str(int(day)+6).zfill(2)
        print(choice,year,month,day_before,day,last_day)
        
        opath='/Users/jason/0_dat/ERA5/events/'+choice+'/'
        # opath='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/data_raw/ERA5/'+choice+'/'
        os.system('mkdir -p '+opath)
        
        ofile=opath+'/'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'-'+last_day+'_3hourly_'+choice+'.grib'
    
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
                        str(int(day)-1).zfill(2),
                        str(day).zfill(2),
                        str(int(day)+1).zfill(2),
                        str(int(day)+2).zfill(2),
                        str(int(day)+3).zfill(2),
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
                        str(int(day)-1).zfill(2),
                        str(day).zfill(2),
                        str(int(day)+1).zfill(2),
                        str(int(day)+2).zfill(2),
                        str(int(day)+3).zfill(2),
                        ],
                    'time': [
                        '00:00', '03:00', '06:00',
                        '09:00', '12:00', '15:00',
                        '18:00', '21:00',
                        ],
                },
                ofile)
        if choice=='tpsft2m':        
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'grib',
                        'variable': ['2m_temperature', 'snowfall', 'total_precipitation',],
                                    'year': year,
                                    'month': str(month).zfill(2),
                                    'day': [
                                        str(int(day)-1).zfill(2),
                                        str(day).zfill(2),
                                        str(int(day)+1).zfill(2),
                                        str(int(day)+2).zfill(2),
                                        str(int(day)+3).zfill(2),
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
                                        str(int(day)-1).zfill(2),
                                        str(day).zfill(2),
                                        str(int(day)+1).zfill(2),
                                        str(int(day)+2).zfill(2),
                                        str(int(day)+3).zfill(2),
                                        ],
                        'time': [
                            '00:00', '03:00', '06:00',
                            '09:00', '12:00', '15:00',
                            '18:00', '21:00',
                        ],
                    },
                ofile)

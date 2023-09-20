#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:24:05 2023
@author: jason box, jeb@geus.dk

gathers CARRA rainfall, snowfall or 2m air temperature

measured elapsed time for each month of data gathering

user:
    defines year range
    uncomments a variable of choice
    specifies path where the CARRA grib file is stored

"""

import cdsapi
import numpy as np
import os
from pathlib import Path
import time
import calendar

# define year range
iyear=1990 ; fyear=2023
iyear=2021 ; fyear=2023
n_years=fyear-iyear+1

years=np.arange(iyear,fyear+1).astype(str)
months=np.arange(1,13).astype(str)

# uncomment a variable of choice
var_choice='rain' ; var_choice2='time_integral_of_rain_flux'
var_choice='snow' ; var_choice2='time_integral_of_total_solid_precipitation_flux'
var_choice='t2m' ; var_choice2='2m_temperature'

# specify path where the CARRA grib file is stored
opath='/Users/jason/0_dat/CARRA_raw/'
opath='/Volumes/LaCie/0_dat/CARRA/CARRA_raw/'

# make folder
os.system('mkdir -p '+opath+var_choice)

for year in years:

    print('year',year)

    for monthx in months:
        month=str(monthx).zfill(2)
        print('month',month)

        if month !='null':
        # if month =='09':
            
            ofile=opath+var_choice+'/'+str(year)+month+'.grib'
            
            print(ofile)
    
            my_file = Path(ofile)
            
            if not(my_file.is_file()):
                
                start_time = time.time()

                print('file does not exist '+ofile)
                
                # set days that vary by month and leap year
                days=[
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30','31']
        
                if month =='02':
                    if calendar.isleap(int(year)):
                        days=[
                            '01', '02', '03',
                            '04', '05', '06',
                            '07', '08', '09',
                            '10', '11', '12',
                            '13', '14', '15',
                            '16', '17', '18',
                            '19', '20', '21',
                            '22', '23', '24',
                            '25', '26', '27',
                            '28','29']
                    else:
                        days=[
                            '01', '02', '03',
                            '04', '05', '06',
                            '07', '08', '09',
                            '10', '11', '12',
                            '13', '14', '15',
                            '16', '17', '18',
                            '19', '20', '21',
                            '22', '23', '24',
                            '25', '26', '27',
                            '28']

                if month =='04' or month =='06' or month =='09' or month =='11':
                    days=[
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30']
                                
                c = cdsapi.Client()

                if var_choice!='t2m':
                    c.retrieve(
                        'reanalysis-carra-single-levels',
                        {
                        'format': 'grib',
                        'domain': 'west_domain',
                        'level_type': 'surface_or_atmosphere',
                        'variable': var_choice2,        
                        'product_type': 'forecast',
                        'time': [
                            '00:00', '12:00',
                        ],
                        'leadtime_hour': '6',
                        'year': str(year),
                        'month': month,
                        'day': days,
                        },
                            ofile)
                else:
                    c.retrieve(
                        'reanalysis-carra-single-levels',
                        {
                            'format': 'grib',
                            'domain': 'west_domain',
                            'level_type': 'surface_or_atmosphere',
                            'variable': '2m_temperature',
                            'product_type': 'analysis',
                            'time': [
                                '00:00', '03:00', '06:00',
                                '09:00', '12:00', '15:00',
                                '18:00', '21:00',
                            ],
                                        'year': str(year),
                                        'month': month,
                                        'day': days,
                        },
                                        ofile)

                elapsed_time = time.time() - start_time
                print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            
            else:
                print('file exists '+ofile)

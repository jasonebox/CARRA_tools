#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 07:28:07 2023

@author: jason
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.basemap import Basemap


iyear=1991 ; fyear=2021
n_years=fyear-iyear+1
years=np.arange(iyear,fyear+1).astype('str')
 

ni=1269 ; nj=1069


# base path
path='/Users/jason/Dropbox/CARRA/CARRA_tools/'
os.chdir(path)


season='ANN'
season='JJA'
# season='DJF'

seasons=['ANN','JJA','DJF']

var_choices=['tp']
# var_choices=['t2m','tp']

stat_type='average'


for var_choice in var_choices:
    for season in seasons:
        print(var_choice,season)
    
        cc=0
        sumx=np.zeros((ni,nj))
    
        if season=='JJA':months=['06','07','08']
        if season=='DJF':months=['01','11','12']
        if season=='ANN':months=['01','02','03','04','05','06','07','08','09','10','11','12']
        for year in years:
            for month in months:
                fn=f'/Users/jason/0_dat/CARRA/output/monthly/{var_choice}/{var_choice}_{year}_{month}_1269x1069.npz'
                # print(fn)
                compressed=np.load(fn)
                # plt.imshow(compressed['carra'])
                # plt.title(year+' '+str(mm+1))
                sumx+=compressed['carra']
                cc+=1
                # asassa
        
        if var_choice=='t2m':
            meanx=sumx/cc
        else:
            meanx=sumx
        
        fn=f'./data/CARRA/{var_choice}_{stat_type}_{season}_{iyear}-{fyear}_{ni}x{nj}.npz'
        b = meanx.astype(np.float16)
        np.savez_compressed(fn,average=b)

        plt.close()
        plt.imshow(meanx)
        plt.title(season)
        plt.colorbar()
        plt.show()
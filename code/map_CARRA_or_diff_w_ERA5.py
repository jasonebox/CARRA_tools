#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jason Box, GEUS, jeb@geus.dk

with inputs from:
    resampling_ERA5_to_CARRA.py
    
comments with !! refer to user-specific variables to possibly change

"""

from netCDF4 import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.basemap import Basemap

iyear=1991 ; fyear=2021
n_years=fyear-iyear+1
years=np.arange(iyear,fyear+1).astype('str')
 

# base path
path='/Users/jason/Dropbox/CARRA/CARRA_tools/'
os.chdir(path)

# read ice mask
fn='./meta/CARRA/CARRA_W_domain_ice_mask.nc'
nc2 = Dataset(fn, mode='r')
# print(nc2.variables)
mask = nc2.variables['z'][:,:]
mask=np.rot90(mask.T)

# plt.imshow(mask)

ni=1269 ; nj=1069

fn='./meta/CARRA/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)
lat=np.rot90(lat.T)

fn='./meta/CARRA/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)
lon=np.rot90(lon.T)


#%% map setup

print('map setup')

th=1
font_size=18
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size

ly='x'

res='l'
if ly=='p':res='h'

# global plot settings
th=1
font_size=18
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
mask_svalbard=1;mask_iceland=1;mask_jan_mayen=1


fn='./meta/CARRA/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)

fn='./meta/CARRA/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)

fn='./meta/CARRA/2.5km_CARRA_west_elev_1269x1069.npy'
elev=np.fromfile(fn, dtype=np.float32)
elev=elev.reshape(ni, nj)

# latx=np.rot90(lat.T)
# lonx=np.rot90(lon.T)
offset=0
lon=lon[offset:ni-offset,offset:nj-offset]
lat=lat[offset:ni-offset,offset:nj-offset]
ni-=offset*2
nj-=offset*2
# print(ni,nj)
LLlat=lat[0,0]
LLlon=lon[0,0]-360
# print("LL",LLlat,LLlon)
# print("UL",lat[ni-1,0],lon[ni-1,0]-360)
lon0=lon[int(round(ni/2)),int(round(nj/2))]-360
lat0=lat[int(round(ni/2)),int(round(nj/2))]
# print("mid lat lon",lat0,lon0)

URlat=lat[ni-1,nj-1]
URlon=lon[ni-1,nj-1]
# print("LR",lat[0,nj-1],lon[0,nj-1]-360)
# print("UR",URlat,URlon)

# m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat, lat_0=72, lon_0=-36, resolution='l', projection='lcc')
m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat, lat_0=lat0, lon_0=lon0, resolution=res, projection='lcc')
# x, y = m(lat, lon)
x, y = m(lon, lat)
#%% map trend


diff_w_ERA=1
diff_name=''
# dx=0.1

var_choices=['tp']; units=" mm y $^{-1}$" ; vmins=[0,0,0] ; vmaxs=[3000,1000,1000] ; dx=[100,0.5,1]
# var_choices=['rf']; units=" mm y $^{-1}$"
# var_choices=['t2m'] ; units=" °C" ; vmins=[-30,-12,-40] ; vmaxs=[10,12,9] ; dx=[1,0.5,1]

# var_choices=['sf']; units=" mm y $^{-1}$"

# var_choices=['rf','tp','t2m']


season='ANN'
season='JJA'
# season='DJF'

seasons=['ANN','JJA','DJF']

# seasons=['JJA']
seasons=['ANN']


for ss,season in enumerate(seasons):

    CARRA_result_grid_path='./data/CARRA/'
    
    stat_type='slope' ; multiplier=31
    stat_type='average' ; multiplier=1
    for var_choice in var_choices:
        th=1
        if var_choice!='sf':
            # fn=CARRA_result_grid_path+var_choice+'_'+stat_type+'_'+str(iyear)+'-'+str(fyear)+'_1269x1069.npy'
            # plotvar=np.fromfile(fn, dtype=np.float16)
            # plotvar=plotvar.reshape(ni, nj)
            fn=f'./data/CARRA/{var_choice}_{stat_type}_{season}_{iyear}-{fyear}_{ni}x{nj}.npz'
            b=np.load(fn)
            plotvar=b['average']
            
            plt.imshow(plotvar)
            plt.title(f'CARRA {season}')
            plt.axis('off')
            plt.colorbar()
            
        else:
            fn='./data/CARRA/tp_slope_'+str(iyear)+'-'+str(fyear)+'_1269x1069.npy'
            plotvara=np.fromfile(fn, dtype=np.float16)
            fn='./data/CARRA/rf_slope_'+str(iyear)+'-'+str(fyear)+'_1269x1069.npy'
            plotvarb=np.fromfile(fn, dtype=np.float16)
            plotvar=plotvara-plotvarb
            plotvar=plotvar.reshape(ni, nj)
        plotvar=plotvar*multiplier
        if diff_w_ERA:
            if var_choice=='t2m':
                vmins=[-5,-5,-5] ; vmaxs=[5,5,5]
            else:
                vmins=[-400,-5,-5] ; vmaxs=[400,5,5]

            # fn='./data/ERA5/resampled/ERA5_'+var_choice+'_'+stat_type+'_'+season+'_'+str(iyear)+"-"+str(fyear)+'_1269x1069.npy'
            # temp=np.fromfile(fn, dtype=np.float16)
            # temp=temp.reshape(ni, nj)
            fn=f'./data/ERA5/resampled/ERA5_{var_choice}_{stat_type}_{season}_{iyear}-{fyear}_{ni}x{nj}.npz'
            b=np.load(fn)
            temp=b['average']
            temp=np.rot90(temp.T)
            plotvar=(temp*multiplier)-plotvar
        # plt.imshow(temp*31)
        # plt.imshow(temp)
        # plt.title(f'ERA5 {season}')
        # plt.axis('off')
        # plt.colorbar()

        # plt.imshow(plotvar)
        # plt.title(f'CARRA {season}')
        # plt.axis('off')
        # plt.colorbar()
        # #%%
        print('map result')
        
        if var_choice=='rf':
            lo=-40 ; hi=-lo 
            lo=-180 ; hi=-lo 
            var_choice2='rainfall'
            maxval=1000
        if var_choice=='tp':
            var_choice2='total precipitation'
            maxval=3000
        if var_choice=='sf':
            lo=-300 ; hi=-lo ; dx=10
            var_choice2='snowfall'
            maxval=6000
        if var_choice=='t2m':
            var_choice2='air temperature'
            maxval=60
            
        plotvar_non_fuzzy=np.rot90(plotvar.T)
        # plotvar_non_fuzzy[mask<0]=0
        
        
        if var_choice!='t2m':
            areax=2.5e3**2
            mass=np.nansum(plotvar[mask>0]*mask[mask>0])/1e9*areax/1000
            
            # mass2=np.nansum(meanx[mask>0]*mask[mask>0])/1e9*areax/1000
        
        
        plt.close()
        fig, ax = plt.subplots(figsize=(12, 12))
        
        do_regional_min_max=1
        
        if do_regional_min_max:
            lonx=lon.copy()
            lonx-=360
            # v=np.where((lonx<-45)&(lat<62))
            maxval=np.nanmax(plotvar_non_fuzzy)
            minval=np.nanmin(plotvar_non_fuzzy)
            alllat=lat[plotvar_non_fuzzy==maxval][0]
            alllon=lonx[plotvar_non_fuzzy==maxval][0]
            minlat=lat[plotvar_non_fuzzy==minval][0]
            minlon=lonx[plotvar_non_fuzzy==minval][0]
            elev_with_min=elev[plotvar_non_fuzzy==minval][0]
            elev_with_max=elev[plotvar_non_fuzzy==maxval][0]
                
        # clevs=np.arange(lo,-lo+dx,dx)
        clevs=np.arange(vmins[ss],vmaxs[ss]+dx[ss],dx[ss])

        z=np.rot90(plotvar.T)
        
        if var_choice=='t2m':
            # cm='bwr'
            cm = plt.cm.seismic
            units2='° C'
            cm.set_under('k')
            cm.set_over('orange')
        
        else:
            # cm='BrBG'
            cm = plt.cm.BrBG
            units2='mm y$^{-1}$'
            units2='mm'
            cm.set_under('r')
            cm.set_over('b')
        pp=m.contourf(x,y,z,clevs,cmap=cm, extend='both')
        
        # do_conf=0
        # confidence_name=''
        
        # if do_conf:
        #     confidence=np.rot90(confidence.T)
        #     z2=z.copy()
        #     z2[confidence>0.66]=np.nan
        #     cs = m.contourf(x, y, z2, hatches=['...'], colors='lightgray',
        #                        extend='both', alpha=0.1)
        #     confidence_name='_66_confidence_mask'
        
        # if plot_only_background==0:
        m.drawcoastlines(color='k',linewidth=0.5)
        # if plot_only_background:m.drawparallels([66.6],color='gray')
        # if plot_only_background:m.drawparallels([60,70,80,83],dashes=[2,4],color='k')
        # m.drawmeridians(np.arange(0.,420.,10.))
        # if plot_only_background:m.drawmeridians([-68,-62,-56,-50,-44,-38,-32,-26,-20])
        # m.drawmapboundary(fill_color='aqua')
        ax = plt.gca()     
        if do_regional_min_max:
            lons, lats = m(lonx, lat)
            m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=400, facecolors='none', edgecolors='k',linewidths=th*2,zorder=29)
            m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=380, facecolors='none', edgecolors='m',linewidths=th*1,zorder=30)
            m.scatter(lons[plotvar_non_fuzzy==minval],lats[plotvar_non_fuzzy==minval], s=400, facecolors='none', edgecolors='k',linewidths=th*2,zorder=29)
            m.scatter(lons[plotvar_non_fuzzy==minval],lats[plotvar_non_fuzzy==minval], s=380, facecolors='none', edgecolors='w',linewidths=th*1,zorder=30)
            
        # --------------------- colorbar
        xx0=1 ; yy0x=0.13 ; dyx=0.45
        cbaxes = fig.add_axes([xx0-0.15, yy0x, 0.015, dyx]) 
        cbar = plt.colorbar(pp,orientation='vertical',format="%d",cax=cbaxes)
        mult=1
        plt.text(xx0+0.01, 0.69,units2, fontsize=font_size*mult,
                 transform=ax.transAxes, color='k') 
        # cbar.ax.minorticks_off()
        
        # cbar.ax.tick_params(length=0)
        
        # from matplotlib.ticker import LogFormatter 
        # formatter = LogFormatter(10, labelOnlyBase=False) 
        
        # cbar = plt.colorbar(pp,orientation='vertical',cax=cbaxes, ticks=bounds, format=formatter)
        # ticks=[1,5,10,20,50]
        # cbar.outline.set_linewidth(.4)  
        # cbar.ax.tick_params(width=.4)
        # mult=0.6
        # cbar.ax.set_yticklabels(bounds, fontsize=font_size*mult)
        
        
        
        # mult=0.8
        # plt.text(0.65,0.8,'rainfall change\n1991 to 2021', fontsize=font_size*mult,
        #          transform=ax.transAxes, color='k') ; cc+=1. 
        
        # plt.colorbar()
        # cc=0
        # xx0=0.0 ; yy0=-0.02 ; dy2=-0.04
        # mult=0.7
        # color_code='grey'
        # plt.text(xx0, yy0+cc*dy2,'Box, Nielsen and the CARRA team', fontsize=font_size*mult,
        #   transform=ax.transAxes,color=color_code) ; cc+=1. 
        
        annotatex=1
        
        if annotatex:
            cc=0
            xx0=1.02 ; yy0=0.98 ; dy2=-0.035
        
            mult=0.95
            color_code='k'
            print()
            diff_name2='CARRA'

            if diff_w_ERA:
                diff_name2='ERA5 minus CARRA'

            plt.text(xx0, yy0+cc*dy2,f'{diff_name2}, {season}', fontsize=font_size*mult,
                     transform=ax.transAxes,color=color_code) ; cc+=1.
            plt.text(xx0, yy0+cc*dy2,var_choice2, fontsize=font_size*mult,
                     transform=ax.transAxes,color=color_code) ; cc+=1.
            plt.text(xx0, yy0+cc*dy2,str(iyear)+' to '+str(fyear), fontsize=font_size*mult,
                     transform=ax.transAxes,color=color_code) ; cc+=2.
                
            
            mult=0.7
            msg="overall max {:.0f}".format(np.max(plotvar))+units
            print(msg)
            plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
              transform=ax.transAxes,color=color_code) ; cc+=1. 
            msg="{:.3f}".format(alllat)+"°N, "+"{:.3f}".format(abs(alllon))+"°W, {:.0f}".format(elev_with_max)+' m'
            print(msg)
            plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
              transform=ax.transAxes,color=color_code) ; cc+=1. 

            msg="overall min {:.0f}".format(np.min(plotvar))+units
            print(msg)
            plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
              transform=ax.transAxes,color=color_code) ; cc+=1. 
            msg="{:.3f}".format(minlat)+"°N, "+"{:.3f}".format(abs(minlon))+"°W, {:.0f}".format(elev_with_min)+' m'
            print(msg)
            plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
              transform=ax.transAxes,color=color_code) ; cc+=2.             
            # plt.text(1.55, 0.99,'.', fontsize=font_size*mult,
            #   transform=ax.transAxes,color='grey') ; cc+=2.         
        
        ly='p'

        # plt.show()

        if ly=='p':
            DPI=100
            if diff_w_ERA:
                diff_name='ERA5_minus'
            fn=f'./Figs/{diff_name}CARRA_{var_choice}_{stat_type}_{season}_{iyear}-{fyear}_{ni}x{nj}.png'
            plt.savefig(fn, bbox_inches='tight', dpi=DPI)
            # os.system('open '+fn)
            
        if ly == 'x':
            plt.show()
        
#%%
make_gif=0

if make_gif:
    print("making gif")
    animpath='/Users/jason/Dropbox/CARRA/CARRA_tools/anim/'    
    import imageio.v2 as imageio

    if make_gif == 1:
        images=[]
        for season in seasons:
            images.append(imageio.imread(f'./Figs/ERA5_minus_CARRA{var_choice}_{stat_type}_{season}_{iyear}-{fyear}_{ni}x{nj}.png'))
        imageio.mimsave(f'{animpath}ERA5_minus_CARRA_t2m_average_seasonal.gif', images,   duration=1000)
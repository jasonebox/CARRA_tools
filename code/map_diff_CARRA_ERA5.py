#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
difference map ERA5 minus CARRA

upstream:
    /Users/jason/Dropbox/CARRA/CARRA_ERA5_events/src/gather_event_3h_from_extremes_list.py
    /Users/jason/Dropbox/CARRA/CARRA_ERA5_events/src/ERA5/Resampling_ERA5_to_CARRA_code_from_extremes_list.py
    /Users/jason/Dropbox/CARRA/CARRA_ERA5_events/src/CARRA/extract_CARRA_rf_from_tpt2m_from_extremes_list.py

updated Dec 2022
@author: Jason Box, GEUS, jeb@geus.dk
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os
from glob import glob
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import pandas as pd
from datetime import datetime 
import numpy as np

if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/'

os.chdir(base_path)


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

ly='x'

res='l'
if ly=='p':res='h'


# casex='2017-09-14'; casex2='14 September, 2017'
# casex='2022-06-17'; casex2='17 June, 2022'

CARRA_or_ERA5='ERA5'
outpath='/Users/jason/0_dat/ERA5/events/resampled/rf/'
fn=outpath+'rf_'+casex+'_1269x1069.npy'
fn=outpath+casex+'_1269x1069.npy'
ERA5=np.fromfile(fn, dtype=np.float16)
ERA5=ERA5.reshape(ni, nj)
ERA5=np.rot90(ERA5.T)

CARRA_or_ERA5='CARRA'
outpath='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/data_raw/'+CARRA_or_ERA5+'/event/'
fn=outpath+'rf_'+casex+'_1269x1069.npy'
CARRA=np.fromfile(fn, dtype=np.float16)
CARRA=CARRA.reshape(ni, nj)
# CARRA=np.rot90(CARRA.T)

do_dif=1

if do_dif:
    diff=ERA5-CARRA
    plotvar=diff
    plotvar_non_fuzzy=diff
    lo=-20 ; hi=-lo
    cm='BrBG'
    model_title='ERA5 minus CARRA'

if do_dif==0: # CARRA
    CARRA_or_ERA5='ERA5'
    # CARRA_or_ERA5='CARRA'
    model_title=CARRA_or_ERA5

    if CARRA_or_ERA5=='ERA5':
        plotvar=ERA5
    if CARRA_or_ERA5=='CARRA':
        plotvar=CARRA
    plotvar_non_fuzzy=CARRA
    lo=0 ; hi=72/2

    # loval=188
    # if i==0:
    loval=255
    # print("loval",loval)
    r=[188,108,76,0,      172,92,0,0,      255,255,255,220, 204,172,140,108, 255,255,255,236, 212,188,164,156, 255, 255]
    g=[255,255,188,124, 255,255,220,156, 255,188,156,124,  156,124,92,60,   188,140,72,0,    148,124,68,28, 255, 255 ]
    b=[255,255,255,255,  172,92,0,0,      172,60,0,0,       156,124,92,60,   220,196,164,0,  255,255,255,196, 0, 255 ]
    r=[loval,108,76,0,      172,92,0,0,      255,255,255,220, 204,172,140,108, 255,255,255,236, 212,188,164,156]
    g=[255,255,188,124, 255,255,220,156, 255,188,156,124,  156,124,92,60,   188,140,72,0,    148,124,68,28 ]
    b=[255,255,255,255,  172,92,0,0,      172,60,0,0,       156,124,92,60,   220,196,164,0,  255,255,255,196]
    colors = np.array([r, g, b]).T / 255
    n_bin = 24
    cmap_name = 'my_list'
    # Create the colormap
    cm = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bin)
    cm.set_under('w') #  brown, land
    # my_cmap.set_over('#E4EEF8')




varnam2=['rainfall','precip.',r'$(t2m_{max} + t2m_{min})/2$']
min_value=[0,0,-30]
max_value=[72,6000,4]
units=['mm','mm','deg C']
 
# read ice mask
fn='./meta/CARRA/CARRA_W_domain_ice_mask.nc'
fn='/Users/jason/Dropbox/CARRA/CARRA_rain/ancil/CARRA_W_domain_elev.nc'
nc2 = Dataset(fn, mode='r')
# print(nc2.variables)
mask = nc2.variables['z'][:,:]
# mask = np.rot90(mask.T)
# plt.imshow(mask)

mask_svalbard=1;mask_iceland=1;mask_jan_mayen=1

plt.close()
fig, ax = plt.subplots(figsize=(12, 12))

# ly='x'
map_version=1
if map_version:
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
    x, y = m(lat, lon)

i=0
# mask_canada=1

# if i<3:
#     if mask_iceland:
#         mask[((lon-360>-30)&(lat<66.6))]=0
#     if mask_svalbard:
#         mask[((lon-360>-20)&(lat>70))]=0
#     if mask_canada:
#         mask[((lon-360<-55)&(lat<73))]=0


plotvar_masked=plotvar.copy()
# plotvar_masked[elevx<800]=np.nan
# plotvar_masked[mask<1]=np.nan

# plotvar*=mask
# plotvar[mask<=0]=np.nan
plotvar_non_fuzzy[mask<=10]=np.nan
# if i==2:
#     plotvar_non_fuzzy[mask==0]=np.nan
#     plotvar[mask==0]=np.nan

# areax=2.5e3**2
# mass=np.sum(plotvar[mask>0]*mask[mask>0])/1e9*areax/1000

# if i<2:plotvar_non_fuzzy[mask==0]=-1

# plt.imshow(lon)
ax = plt.subplot(111)
# tit=year+' CARRA '+varnam2[i]+' over Greenland ice'
# ax.set_title(tit)

if map_version==0:
    pp=plt.imshow(plotvar_non_fuzzy, interpolation='nearest', origin='lower', cmap=cm,vmin=lo,vmax=hi) ; plt.axis('off') 

lon-=360
# if i<2:
#     v=np.where((lon<-45)&(lat<62))
#     maxval=np.max(plotvar_non_fuzzy)
#     minval=np.min(plotvar_non_fuzzy[plotvar_non_fuzzy>1])
#     maxval2=np.max(plotvar_non_fuzzy[v])
#     # print("position of extremes")
#     # print(np.max(plotvar_non_fuzzy))
#     # print(maxval)
#     # print(lat[plotvar==maxval2])
#     # print(lon[plotvar==maxval2])
#     SSWlat=lat[plotvar_non_fuzzy==maxval2][0]
#     SSWlon=lon[plotvar_non_fuzzy==maxval2][0]
#     elev_atSSW=elev[plotvar_non_fuzzy==maxval2][0]
#     # print(lat[plotvar==maxval2])
#     # print(lon[plotvar==maxval2])
#     alllat=lat[plotvar_non_fuzzy==maxval][0]
#     alllon=lon[plotvar_non_fuzzy==maxval][0]
#     minlat=lat[plotvar_non_fuzzy==minval][0]
#     minlon=lon[plotvar_non_fuzzy==minval][0]

if map_version:
    pp=m.imshow(plotvar, cmap=cm,vmin=lo,vmax=hi) 
    # m.axis('off')
    m.drawcoastlines(color='k',linewidth=0.5)
    # m.drawparallels([66.6],color='gray')
    # m.drawparallels([60,70,80],dashes=[2,4],color='k')
    # m.drawmeridians(np.arange(0.,420.,10.))
    # m.drawmapboundary(fill_color='aqua')
    ax = plt.gca()     
    # plt.title("Lambert Conformal Projection")
    # plt.show()
    lons, lats = m(lon, lat)
    # m.scatter(lons[plotvar_non_fuzzy==maxval2],lats[plotvar_non_fuzzy==maxval2], s=380, facecolors='none', edgecolors='m')
    # m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=400, facecolors='none', edgecolors='k',linewidths=th*2)
    # m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=380, facecolors='none', edgecolors='m',linewidths=th*1)
    # m.scatter(lons[plotvar_non_fuzzy==minval],lats[plotvar_non_fuzzy==minval], s=380, facecolors='none', edgecolors='m')
    # m.contour(lons, lats,plotvar_masked,[0.1],linewidths=1,colors='lightgrey')
              
if i<2: # rf or tp
    cbar_min=lo
    cbar_max=-lo
    cbar_step=max_value[i]/24
    cbar_num_format = "%d"

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    # plt.colorbar(pp)            
    cbar = plt.colorbar(pp,
                        orientation='vertical',format=cbar_num_format, cax=cax)
    cbar.ax.set_ylabel(units[i], fontsize = font_size)
    # tickranges=np.arange(cbar_min, cbar_max+cbar_step, cbar_step).astype(int)
    # # print(tickranges)
    # cbar.ax.set_yticklabels(tickranges, fontsize=font_size)
    # clb=
    # plt.colorbar()
    # clb.ax.tick_params(labelsize=font_size) 
    # clb.ax.set_title(units,fontsize=font_size)
    
    # clb=plt.colorbar()
    # clb.ax.tick_params(labelsize=font_size) 
    # clb.ax.set_title(units,fontsize=font_size)

if i==2: # t2m
    cbar_min=min_value[i]
    cbar_max=max_value[i]
    cbar_step=max_value[i]/24
    cbar_num_format = "%d"

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    # plt.colorbar(im)            
    cbar = plt.colorbar(pp,orientation='vertical',format=cbar_num_format, cax=cax)
    cbar.ax.set_ylabel(units[i], fontsize = font_size)
    # tickranges=np.arange(cbar_min, cbar_max+cbar_step, cbar_step)
    # # print(tickranges)
    # cbar.ax.set_yticklabels(tickranges, fontsize=font_size)

# cc=0
# xx0=0.0 ; yy0=-0.02 ; dy2=-0.04
# mult=0.7
# color_code='grey'
# plt.text(xx0, yy0+cc*dy2,'Box, Nielsen and the CARRA team', fontsize=font_size*mult,
#   transform=ax.transAxes,color=color_code) ; cc+=1. 


if i<2:
    cc=0
    xx0=0.44 ; yy0=0.17 ; dy2=-0.028
    xx0=0.34 ; yy0=0.5 ; dy2=-0.028
    xx0=0.35 ; yy0=0.68 ; dy2=-0.028
    mult=0.8
    color_code='b'
    print()
    plt.text(xx0, yy0+cc*dy2,'rainfall difference\n'+model_title+'\n'+casex2, fontsize=font_size*mult,
      transform=ax.transAxes,color=color_code) ; cc+=1.

    # msg="{:.1f}".format(mass)+" Gt "+varnam2[i]+" total mass flux"
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 

    # msg="{:.0f}".format(np.max(plotvar))+" mm "+"max "+varnam2[i]+""
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 

    # msg="{:.4f}".format(alllat)+" N, "+"{:.4f}".format(abs(alllon))+" W"
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 
    
    # msg="{:.0f}".format(np.max(plotvar[v]))+" mm / y "+"max "+varnam2[i]+" SSW"
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 

    # msg="{:.4f}".format(SSWlat)+" N, "+"{:.4f}".format(abs(SSWlon))+" W"
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 

if ly == 'x':
    plt.show()


if ly =='p':
    figpath='/Users/jason/Dropbox/CARRA/CARRA rainfall study/Figs/'
    figpath='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/Figs/'
    # os.system('mkdir -p '+figpath)
    # figpath='./Figs/annual/'+varnams[i]+'/'+str(DPI)+'/'
    # os.system('mkdir -p '+figpath)
    if do_dif:
        figname=figpath+'rf_ERA5_minus_CARRA_'+casex
    else:
        figname=figpath+'rf_'+CARRA_or_ERA5+'_'+casex
    # if i<2:
    plt.savefig(figname+'.png', bbox_inches='tight')
    # plt.savefig(figname+'.pdf', bbox_inches='tight')
    # else:
    #     plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)#, facecolor=fig.get_facecolor(), edgecolor='none')


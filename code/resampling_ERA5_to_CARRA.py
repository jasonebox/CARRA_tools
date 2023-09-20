# -*- coding: utf-8 -*-
"""

resamples ERA5 data to CARRA grid

is designed for CARRA west domain, can be changed to east domain given correct meta data

comments with !! refer to user-specific variables to possibly change

"""
import numpy as np
import pandas as pd
import os
from pyproj import Proj, transform
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# !! base path, change to your local path
path='/Users/jason/Dropbox/CARRA/CARRA_tools/'
os.chdir(path)

# function to adjust longitude to be negative when west of prime meridian
def lon360_to_lon180(lon360):
    # reduce the angle  
    lon180 =  lon360 % 360 
    # force it to be the positive remainder, so that 0 <= angle < 360  
    lon180 = (lon180 + 360) % 360;  
    # force into the minimum absolute value residue class, so that -180 < angle <= 180  
    lon180[lon180 > 180] -= 360
    return lon180

# CARRA West grid dimensions
ni = 1269 ; nj = 1069

# read CARRA lat lon arrays
fn = './meta/CARRA/2.5km_CARRA_west_lat_1269x1069.npy'
lat = np.fromfile(fn, dtype=np.float32)
clat_mat = lat.reshape(ni, nj)

fn = './meta/CARRA/2.5km_CARRA_west_lon_1269x1069.npy'
lon = np.fromfile(fn, dtype=np.float32)
lon_pn = lon360_to_lon180(lon)
clon_mat = lon_pn.reshape(ni, nj) 


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

# %% load ERA5 coordinates 

#ERA 5 elevation data
fn='./meta/ERA5/ERA5_mask_and_terrain.nc'
ds=xr.open_dataset(fn)

# projections, from lat/lon (inProj) to meters (outProj)
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



#%% load ERA5 dataset
var_choice='sf'
var_choice='t2m'
# var_choice='tp'
iyear=1979 ; fyear=2022
iyear=1991 ; fyear=2021

stat_type='slope' ; stat_type_name='stats'
# stat_type='stats' ; stat_type_name='slope'
stat_type='average' ; stat_type_name='average'

# fn='/Users/jason/Dropbox/CARRA/CARRA_tools/data/ERA5/ERA5_'+var_choice+"_mean_"+str(iyear)+"-"+str(fyear)+".npy"
# temp = np.fromfile(fn)#, dtype=np.float16)
# temp = temp.reshape(720,1440)

opath='/Users/jason/Dropbox/CARRA/CARRA_tools/data/'
opath="/Users/jason/Dropbox/ERA5/output/"
fn=opath+var_choice+"_"+stat_type+"_"+str(iyear)+"-"+str(fyear)+".nc"
# fn="/Users/jason/Dropbox/ERA5/output/"+var_choice+"_"+stat_type_name+"_"+str(iyear)+"-"+str(fyear)+".nc"
ds=xr.open_dataset(fn)
ds.variables
list(ds.keys())
temp=np.array(ds.variables[stat_type][:,:,:])
np.shape(temp)
#  visualisation
plt.close()
plt.imshow(temp[1,:,:])#,vmin=0,vmax=12)
plt.axis('off')
# plt.title(var_choice+' '+date_stringx)
# plt.colorbar()


#%% map setup

print('map setup')
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
#%%
season='ANN'
season='JJA'
# season='DJF'

seasons=['ANN','JJA','DJF']
# seasons=['ANN']
# seasons=['JJA']

for ss,season in enumerate(seasons):
    if var_choice=='sf':
        if season=='ANN':
            ERA_data=np.nansum(temp,axis=0)
        if season=='JJA':
            ERA_data=np.nansum(temp[5:7],axis=0)
        if season=='DJF':
            ERA_data=np.nansum(temp[[0,10,11]],axis=0)
    if var_choice=='tp':
        if season=='ANN':
            ERA_data=np.nansum(temp,axis=0)
        if season=='JJA':
            ERA_data=np.nansum(temp[5:7],axis=0)
        if season=='DJF':
            ERA_data=np.nansum(temp[[0,10,11]],axis=0)
    if var_choice=='t2m':
        if season=='ANN':
            ERA_data=np.nanmean(temp,axis=0)
        if season=='JJA':
            ERA_data=np.nanmean(temp[5:7],axis=0)
        if season=='DJF':
            ERA_data=np.nanmean(temp[[0,10,11]],axis=0)
    
    np.shape(ERA_data)
    
    # plt.close()
    # plt.imshow(ERA_data)#,vmin=0,vmax=12)
    # plt.axis('off')
    # plt.title('ERA_data')
    # plt.colorbar()
    
    # print(ds.variables)
    # print(np.shape(ds.tp))
    ##%% resampling ERA to CARRA
    
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
    
    outpath='/Users/jason/Dropbox/CARRA/CARRA_tools/data/ERA5/resampled/'
    os.system('mkdir -p '+outpath)
    
    new_grid= ERA_data[CARRA_cells_for_ERA5.col_e, CARRA_cells_for_ERA5.row_e]
    
    # CARRA domain coordinates
    ni=1269 ; nj=1069
    # new_grid = np.rot90(new_grid.reshape(ni, nj).T)
    new_grid = new_grid.reshape(ni, nj)
    
    # fn='./data/ERA5/resampled/ERA5_'+var_choice+'_'+stat_type+'_'+season+'_'+str(iyear)+"-"+str(fyear)+'_1269x1069.npy'
    # new_grid.astype(dtype=np.float16).tofile(fn)
    
    fn=f'./data/ERA5/resampled/ERA5_{var_choice}_{stat_type}_{season}_{iyear}-{fyear}_{ni}x{nj}.npz'
    b = new_grid.astype(np.float16)
    np.savez_compressed(fn,average=b)

    do_plot=1
    
    if do_plot:
        #  visualisation
        vmins=[-30,-12,-40] ; vmaxs=[10,9,9] ; dx=[1,0.5,1]

        # global plot settings
        th=1
        font_size=18
        plt.rcParams['axes.facecolor'] = 'k'
        plt.rcParams['axes.edgecolor'] = 'k'
        plt.rcParams["font.size"] = font_size
        plt.rcParams['axes.facecolor']='w'
        plt.rcParams['savefig.facecolor']='w'
        mult=1
        if var_choice=='t2m':
            # cm='bwr'
            cm = plt.cm.seismic
            units2='° C'
            cm.set_under('k')
            cm.set_over('orange')
        else:
            cmap='BrBG'
        plt.close()
        fig, ax = plt.subplots(figsize=(12, 12))
        
        do_regional_min_max=1
        
        if do_regional_min_max:
            fn='./meta/CARRA/2.5km_CARRA_west_lat_1269x1069.npy'
            lat=np.fromfile(fn, dtype=np.float32)
            lat=lat.reshape(ni, nj)
            lat=np.rot90(lat.T)

            fn='./meta/CARRA/2.5km_CARRA_west_lon_1269x1069.npy'
            lon=np.fromfile(fn, dtype=np.float32)
            lon=lon.reshape(ni, nj)
            lon=np.rot90(lon.T)
            lonx=lon.copy()
            lonx-=360
            # v=np.where((lonx<-45)&(lat<62))
            # plotvar_non_fuzzy=new_grid
            plotvar_non_fuzzy=np.rot90(new_grid.T)

            plotvar=new_grid
            maxval=np.nanmax(plotvar_non_fuzzy)
            minval=np.nanmin(plotvar_non_fuzzy)
            alllat=lat[plotvar_non_fuzzy==maxval][0]
            alllon=lonx[plotvar_non_fuzzy==maxval][0]
            minlat=lat[plotvar_non_fuzzy==minval][0]
            minlon=lonx[plotvar_non_fuzzy==minval][0]
            elev_with_min=elev[plotvar_non_fuzzy==minval][0]
            elev_with_max=elev[plotvar_non_fuzzy==maxval][0]

        clevs=np.arange(vmins[ss],vmaxs[ss]+dx[ss],dx[ss])
        # pp=plt.imshow(new_grid,cmap=cmap,vmin=vmins[ss],vmax=vmaxs[ss])
        pp=m.contourf(x,y,new_grid,clevs,cmap=cm, extend='both')
        m.drawcoastlines(color='k',linewidth=0.5)

        plt.axis('off')
        # plt.title('ERA5 resampled to CARRA grid\n'+var_choice+' '+season+' '+stat_type+' '+str(iyear)+"-"+str(fyear))
        # cbar=plt.colorbar()
        # cbar.ax.set_title('°C')
        xx0=1.02 ; yy0=0.98 ; dy2=-0.035 ; cc=0 ; color_code='k'

        plt.text(xx0, yy0+cc*dy2,f'ERA5, {season}', fontsize=font_size*mult,
                 transform=ax.transAxes,color=color_code) ; cc+=1.
        plt.text(xx0, yy0+cc*dy2,var_choice, fontsize=font_size*mult,
                 transform=ax.transAxes,color=color_code) ; cc+=1.
        plt.text(xx0, yy0+cc*dy2,str(iyear)+' to '+str(fyear), fontsize=font_size*mult,
                 transform=ax.transAxes,color=color_code) ; cc+=2.
        
        mult=0.7
        units='°C'
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
        
        # ax = plt.gca()     
        if do_regional_min_max:
            fn='./meta/CARRA/2.5km_CARRA_west_lat_1269x1069.npy'
            lat=np.fromfile(fn, dtype=np.float32)
            lat=lat.reshape(ni, nj)
            lat=np.rot90(lat.T)

            fn='./meta/CARRA/2.5km_CARRA_west_lon_1269x1069.npy'
            lon=np.fromfile(fn, dtype=np.float32)
            lon=lon.reshape(ni, nj)
            lon=np.rot90(lon.T)
            lonx=lon.copy()
            lonx-=360
            lons, lats = m(lonx, lat)
            m.scatter(lons[plotvar_non_fuzzy==maxval][0],lats[plotvar_non_fuzzy==maxval][0], s=400, facecolors='none', edgecolors='k',linewidths=th*2,zorder=29)
            m.scatter(lons[plotvar_non_fuzzy==maxval][0],lats[plotvar_non_fuzzy==maxval][0], s=380, facecolors='none', edgecolors='m',linewidths=th*1,zorder=30)
            m.scatter(lons[plotvar_non_fuzzy==minval][0],lats[plotvar_non_fuzzy==minval][0], s=400, facecolors='none', edgecolors='k',linewidths=th*2,zorder=29)
            m.scatter(lons[plotvar_non_fuzzy==minval][0],lats[plotvar_non_fuzzy==minval][0], s=380, facecolors='none', edgecolors='w',linewidths=th*1,zorder=30)
            
        # --------------------- colorbar
        xx0=1 ; yy0x=0.13 ; dyx=0.45
        cbaxes = fig.add_axes([xx0-0.15, yy0x, 0.015, dyx]) 
        cbar = plt.colorbar(pp,orientation='vertical',format="%d",cax=cbaxes)
        mult=1
        plt.text(xx0+0.01, 0.69,'°C', fontsize=font_size*mult,
                 transform=ax.transAxes, color='k') 

        # plt.show()                    
        # plt.text(1.3, 1.01,'.', fontsize=font_size*mult,
        #   transform=ax.transAxes,color='grey') 
        # plt.text(1.3, -0.06,'.', fontsize=font_size*mult,
        #   transform=ax.transAxes,color='grey') 
        DPI=100
        ly='p'
        if ly=='x':
            plt.show()                    
        if ly == 'p':
            figpath='./Figs/'
            os.system('mkdir -p '+figpath)
            plt.savefig(figpath+'ERA5_'+var_choice+'_'+stat_type+'_'+season+'_'+str(iyear)+"-"+str(fyear)+'_resampled_to_CARRA_grid.png', bbox_inches='tight', pad_inches=0.04, dpi=DPI, facecolor='w', edgecolor='k')


#%%
make_gif=0

if make_gif:
    print("making gif")
    animpath='/Users/jason/Dropbox/ERA5/anim/'    
    import imageio.v2 as imageio

    if make_gif == 1:
        images=[]
        for season in seasons:
            images.append(imageio.imread(f'/Users/jason/Dropbox/CARRA/CARRA_tools/Figs/ERA5_t2m_average_{season}_1991-2021_resampled_to_CARRA_grid.png'))
        imageio.mimsave(f'{animpath}ERA5_t2m_average_seasonal.gif', images,   duration=1000)

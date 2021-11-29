# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:14:22 2020

@author: amton
"""

import os
os.environ['PROJ_LIB'] = r"C:\Users\cijzendoornvan\AppData\Local\Continuum\anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share"
# import mpl_toolkits.basemap
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
#

plt.close('all')

outputdir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Figures/Figure4_maps/"

#%% Plot zoom

fig = plt.figure(figsize=(12,9))

# m = Basemap(projection='moll',
#            llcrnrlat = 50,
#            urcrnrlat = 54,
#            llcrnrlon = 0,
#            urcrnrlon = 10,
#            resolution = 'f')

m = Basemap(projection='merc',
            llcrnrlat=50.7,
            urcrnrlat=53.7,
            llcrnrlon=3.25,
            urcrnrlon=7.4,
            lat_ts=20,
            resolution='h') #c = crude, h = high, f = full, i = 

m.drawcoastlines()
m.drawcountries(color='black',linewidth=1)
# m.drawstates(color='blue')
# m.drawcounties(color='orange')
m.drawrivers(color='grey',linewidth=0.15)

m.drawmapboundary(color='white', linewidth=1, fill_color='white')
m.fillcontinents(color='white') #lake_color='white'

m.drawlsmask(land_color='lightgreen', ocean_color='white', lakes=True)

m.drawmapscale(3.6, 53.5, 4, 52, 50)

# m.etopo()
# m.bluemarble()
# m.shadedrelief()

# m.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
# m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])

# np.arange(start,stop,step)
# labels=[left,right,top,bottom]

# plt.title('Basemap tutorial', fontsize=20)

plt.show()

plt.savefig(outputdir + 'map_nederland', bbox_inches = 'tight', dpi=300)

#%% Plot zoom

fig = plt.figure(figsize=(12,9))

# m = Basemap(projection='moll',
#            llcrnrlat = 50,
#            urcrnrlat = 54,
#            llcrnrlon = 0,
#            urcrnrlon = 10,
#            resolution = 'f')

m = Basemap(projection='merc',
            llcrnrlat=22.8,
            urcrnrlat=52.,
            llcrnrlon=-140.,
            urcrnrlon=-55.,
            lat_ts=20,
            resolution='l') #c = crude, h = high, f = full, i = intermediate, l = low

m.drawcoastlines()
m.drawcountries(color='grey',linewidth=1)
m.drawstates(color='lightgrey')
# m.drawcounties(color='orange')
# m.drawrivers(color='blue',linewidth=0.15)

m.drawmapboundary(color='white', linewidth=1, fill_color='white')
m.fillcontinents(color='white') #lake_color='white'

m.drawlsmask(land_color='lightgreen', ocean_color='white', lakes=True)

m.drawmapscale(-65, 35, -120, 38, 1000)

# m.etopo()
# m.bluemarble()
# m.shadedrelief()

# m.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
# m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])

# np.arange(start,stop,step)
# labels=[left,right,top,bottom]

# plt.title('Basemap tutorial', fontsize=20)

plt.show()

plt.savefig(outputdir + 'map_unitedstates', bbox_inches = 'tight', dpi=300)


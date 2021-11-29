# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:47:09 2021

@author: cijzendoornvan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata
from netCDF4 import Dataset, num2date

# Create functions
def file2df(file_name):
    points = pd.read_csv(file_name, sep='\t', header=None)
    points = points.dropna(axis=1, how='all')
    points.columns = ["name", "x", "y", "z"]
    points = points.set_index('name')
    return points

import statsmodels.api as sm
def fit_line2(x, y):
    """Return slope, intercept of best fit line."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X, missing='drop') # ignores entires where x or y is NaN
    fit = model.fit()
    # rsquared = fit.rsquared
    return fit.params[1], fit.params[0]#, rsquared # could also return stderr in each via fit.bse

def get_interp_line (x_coord, y_coord, minx, maxx):    # detemine best fit line through transect coordinates
    slope, intercept = fit_line2(x_coord, y_coord)
    
    # create new coordinates for interpolation
    xnew = np.linspace(minx, maxx)
    ynew = [(slope*x + intercept) for x in xnew]
    
    # calculate distance from first point (seaward) for new coordinates
    dist_new = []
    for i, xn in enumerate(xnew):
        dist_new.append(np.sqrt((xnew[-1]-xn)**2 + (ynew[-1]-ynew[i])**2))   
    
    return xnew, ynew, dist_new, intercept, slope

def get_dist(x_coord, y_coord, intercept, slope, xnew, ynew):   
    slope_p = (-1.0 / slope) # get slope for line perpendicular to best fit line
    # get distance of coordinates projected onto best fit line for each transect coordinate 
    dist_proj = []
    for i, xc in enumerate(x_coord):   
        intercept_p = y_coord[i] - slope_p*xc # get intercept for line perpendicular to best fit line that goes through transect coordinate
        x_proj = (intercept_p - intercept)/(slope - slope_p)
        y_proj = (slope*x_proj + intercept)
        dist_proj.append(np.sqrt((xnew[0]-x_proj)**2 + (ynew[0]-y_proj)**2))
    return dist_proj

def get_interpolation(x, elev, xnew):
    # interpolate elevation of projected transect coordinates 
    f = interp1d(x, elev, bounds_error = False)    
    elev_new = f(xnew)  
    return elev_new

#%% Set directories

inputdir = 'C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Data/'
outputdir = 'C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Figures/'

#%% Profiles Noordwijk 5 February 2020
# Load vertical sampling GPS data
filename_0502 = inputdir + '/Noordwijk/GPS/Noordwijk_GPS_05022020.txt'
points_0502 = file2df(filename_0502)
selection = [d for d in points_0502.index if 'V' not in d]
points_0502_trsct = points_0502.loc[selection]

# Sampling locations are indicated with V
sample_points_0502 = points_0502.loc[[d for d in points_0502.index if 'V' in d]]
sample_points_0502_HW = points_0502.loc[[d for d in points_0502.index if 'TRSCT25 WRKL' in d]]
# comparable high water elevations, so wreckline (WRKL) of previous high tide is taken as HW line for next high water

# Split up transect points before and after high water
points_0502_trsct1 = points_0502_trsct.iloc[0:29]
points_0502_trsct2 = points_0502_trsct.iloc[29:]

# create best fit line between the two transects
xnew_0502, ynew_0502, dist_new_0502, intercept_0502, slope_0502 = get_interp_line(points_0502_trsct.x, points_0502_trsct.y, int(min(points_0502_trsct.x)), max(points_0502_trsct.x))

# project the measured transects on the best fit line and calculate the interpolated elevation and cross-shore distance
dist_proj_0502_1 = get_dist(points_0502_trsct1.x, points_0502_trsct1.y, intercept_0502, slope_0502, xnew_0502, ynew_0502)
elev_0502_1 = get_interpolation(dist_proj_0502_1, points_0502_trsct1.z, dist_new_0502)

dist_proj_0502_2 = get_dist(points_0502_trsct2.x, points_0502_trsct2.y, intercept_0502, slope_0502, xnew_0502, ynew_0502)
elev_0502_2 = get_interpolation(dist_proj_0502_2, points_0502_trsct2.z, dist_new_0502)

# Calculate distance and elevation for sample locations
sample_points_0502['dist'] = get_dist(sample_points_0502.x, sample_points_0502.y, intercept_0502, slope_0502, xnew_0502, ynew_0502)
sample_points_0502['elev_proj'] = get_interpolation(dist_proj_0502_1, points_0502_trsct1.z.values, sample_points_0502['dist'])

sample_points_0502_HW['dist'] = get_dist(sample_points_0502_HW.x, sample_points_0502_HW.y, intercept_0502, slope_0502, xnew_0502, ynew_0502)
sample_points_0502_HW['elev_proj'] = get_interpolation(dist_proj_0502_1, points_0502_trsct1.z.values, sample_points_0502_HW['dist'])

# Plot profile
fig1, ax1 = plt.subplots(figsize=(11, 3.5))
fig1.subplots_adjust(bottom=0.2)

plt.plot(dist_new_0502, elev_0502_1, 'k', linewidth = 2.5, label = 'Before high water')
plt.plot(dist_new_0502, elev_0502_2, 'grey', linewidth = 2.5, label = 'After high water')
plt.hlines(sample_points_0502_HW.elev_proj, 0, sample_points_0502_HW.dist, color='royalblue', linewidth=2, label = 'High water line')
plt.scatter(sample_points_0502.iloc[0:3].dist, sample_points_0502.iloc[0:3].elev_proj+0.08, color='k', marker='v', linewidth=2, zorder=10, s=100, label='Sample location')

plt.xlim([0, 110])
plt.ylim([-0.5, 2.5])
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.set_xlabel('Cross-shore distance (m)', fontsize=16)
ax1.set_ylabel('Elevation (m)', fontsize=16)
locs = ['1','2', '3']
for i, txt in enumerate(locs):
    ax1.annotate(txt, (sample_points_0502.iloc[0:3].dist[i]-0.85, sample_points_0502.iloc[0:3].elev_proj[i]+0.28), fontsize=16)

leg = plt.legend(fontsize=16, loc='upper left', framealpha=1)

file_name = 'Noordwijk_profile_050220'
plt.savefig(outputdir + '/CoastalProfiles/' + file_name + '.png')
# plt.close()

#%% Profiles Noordwijk 20 December 2020
# Load vertical sampling GPS data
filename_2012 = inputdir + '/Noordwijk/GPS/Noordwijk_GPS_20122020.txt'
points_2012 = file2df(filename_2012)
selection = [d for d in points_2012.index if 'GPS' in d]
points_2012_trsct = points_2012.loc[selection]

# Sampling locations are indicated with MEAS, high water line with HW
sample_points_2012 = points_2012.loc[[d for d in points_2012.index if 'MEAS' in d]]
sample_points_2012_HW = points_2012.loc[[d for d in points_2012.index if 'HW' in d]]

# Split up transect points before and after high water
points_2012_trsct1 = points_2012_trsct.iloc[0:29]
points_2012_trsct2 = points_2012_trsct.iloc[29:]

# create best fit line between the two transects
xnew_2012, ynew_2012, dist_new_2012, intercept_2012, slope_2012 = get_interp_line(points_2012_trsct.x, points_2012_trsct.y, int(min(points_2012_trsct.x)), max(points_2012_trsct.x))

# project the measured transects on the best fit line and calculate the interpolated elevation and cross-shore distance
dist_proj_2012_1 = get_dist(points_2012_trsct1.x, points_2012_trsct1.y, intercept_2012, slope_2012, xnew_2012, ynew_2012)
elev_2012_1 = get_interpolation(dist_proj_2012_1, points_2012_trsct1.z, dist_new_2012)

dist_proj_2012_2 = get_dist(points_2012_trsct2.x, points_2012_trsct2.y, intercept_2012, slope_2012, xnew_2012, ynew_2012)
elev_2012_2 = get_interpolation(dist_proj_2012_2, points_2012_trsct2.z, dist_new_2012)

# Calculate distance and elevation along best fit line for sample locations
sample_points_2012['dist'] = get_dist(sample_points_2012.x, sample_points_2012.y, intercept_2012, slope_2012, xnew_2012, ynew_2012)
sample_points_2012['elev_proj'] = get_interpolation(dist_proj_2012_1, points_2012_trsct1.z.values, sample_points_2012['dist'])

sample_points_2012_HW['dist'] = get_dist(sample_points_2012_HW.x, sample_points_2012_HW.y, intercept_2012, slope_2012, xnew_2012, ynew_2012)
sample_points_2012_HW['elev_proj'] = get_interpolation(dist_proj_2012_1, points_2012_trsct1.z.values, sample_points_2012_HW['dist'])

# Plot profile
fig1 = plt.subplots(figsize=(12, 4))

plt.plot(dist_new_2012, elev_2012_1, 'k', linewidth = 2.5, label = 'before high water')
plt.plot(dist_new_2012, elev_2012_2, 'grey', linewidth = 2.5, label = 'after high water')
plt.scatter(sample_points_2012.dist, sample_points_2012.elev_proj+0.08, color='k', marker='v', linewidth=2, zorder=10, s=100)
plt.hlines(sample_points_2012_HW.elev_proj, 0, sample_points_2012_HW.dist, color='royalblue', linewidth=2)
plt.xlim([0, 110])
plt.ylim([-0.5, 2])

leg = plt.legend(fontsize=16, loc='lower right')

file_name = 'Noordwijk_profile_201220'
plt.savefig(outputdir + '/CoastalProfiles/' + file_name + '.png')
# plt.close()

#%% Profiles Noordwijk 21 January 2021
# Load vertical sampling GPS data
filename_2101_1 = inputdir + '/Noordwijk/GPS/Noordwijk_GPS_21012021_1.txt'
filename_2101_2 = inputdir + '/Noordwijk/GPS/Noordwijk_GPS_21012021_2.txt'

points_2101_1 = file2df(filename_2101_1)
# get poitns of first transect
selection = [d for d in points_2101_1.index if 'HW' not in d and 'GPS0001' not in d and 'GPS0027' not in d and 'GPS0028' not in d]
points_2101_trsct1 = points_2101_1.loc[selection]

points_2101_part2 = file2df(filename_2101_2)
# get points of second transect
selection = [d for d in points_2101_part2.index if 'GPSS' not in d and 'HW' not in d and 'EP' not in d 
             and 'GPS00' not in d and 'GPS030' not in d and 'GPS0001' not in d and 'GPS0027' not in d and 'GPS0028' not in d
             and 'GPS048' not in d and 'GPS049' not in d and 'GPS050' not in d]
points_2101_trsct2 = points_2101_part2.loc[selection]

# Get erosion pin locations indicated by EP
selection = [d for d in points_2101_part2.index if 'EP' in d]
points_2101_eropin = points_2101_part2.loc[selection]

# create best fit line between the two transects
points_2101_all = pd.concat([points_2101_trsct1, points_2101_trsct2])
xnew_2101, ynew_2101, dist_new_2101, intercept_2101, slope_2101 = get_interp_line(points_2101_trsct1.x, points_2101_trsct1.y, int(min(points_2101_all.x)), max(points_2101_all.x))

# project the measured transects on the best fit line and calculate the interpolated elevation and cross-shore distance
dist_proj_2101_1 = get_dist(points_2101_trsct1.x, points_2101_trsct1.y, intercept_2101, slope_2101, xnew_2101, ynew_2101)
dist_proj_2101_2 = get_dist(points_2101_trsct2.x, points_2101_trsct2.y, intercept_2101, slope_2101, xnew_2101, ynew_2101)

elev_2101_1 = get_interpolation(dist_proj_2101_1, points_2101_trsct1.z, dist_new_2101)
elev_2101_2 = get_interpolation(dist_proj_2101_2, points_2101_trsct2.z, dist_new_2101)

# Get sample locations indicated by GPS0001, GPS0027 and GPS0028
sample_points_2101_1 = points_2101_1.loc[[d for d in points_2101_1.index if 'GPS0001' in d or 'GPS0027' in d or 'GPS0028' in d]]
# Get sample locations indicated by GPSS
sample_points_2101_2 = points_2101_part2.loc[[d for d in points_2101_part2.index if 'GPSS' in d]]
# Get high water line indicated by HW
sample_points_2101_HW = points_2101_1.loc[[d for d in points_2101_1.index if 'HW' in d]]

# Calculate distance and elevation along best fit line for sample locations
sample_points_2101_1['dist'] = get_dist(sample_points_2101_1.x, sample_points_2101_1.y, intercept_2101, slope_2101, xnew_2101, ynew_2101)
sample_points_2101_2['dist'] = get_dist(sample_points_2101_2.x, sample_points_2101_2.y, intercept_2101, slope_2101, xnew_2101, ynew_2101)
sample_points_2101_HW['dist'] = get_dist(sample_points_2101_HW.x, sample_points_2101_HW.y, intercept_2101, slope_2101, xnew_2101, ynew_2101)
points_2101_eropin['dist'] = get_dist(points_2101_eropin.x, points_2101_eropin.y, intercept_2101, slope_2101, xnew_2101, ynew_2101)

sample_points_2101_1['elev_proj'] = get_interpolation(dist_proj_2101_1, points_2101_trsct1.z.values, sample_points_2101_1['dist'])
sample_points_2101_2['elev_proj'] = get_interpolation(dist_proj_2101_2, points_2101_trsct2.z.values, sample_points_2101_2['dist'])
sample_points_2101_HW['elev_proj'] = get_interpolation(dist_proj_2101_1, points_2101_trsct1.z.values, sample_points_2101_HW['dist'])
points_2101_eropin['elev_proj'] = get_interpolation(dist_proj_2101_2, points_2101_trsct2.z.values, points_2101_eropin['dist'])

# Plot profile
fig2, ax2 = plt.subplots(figsize=(11, 3.5))
fig2.subplots_adjust(bottom=0.2)

plt.plot(dist_new_2101, elev_2101_1, 'k', linewidth = 2.5, label = 'Before high water')
plt.plot(dist_new_2101, elev_2101_2, 'grey', linewidth = 2.5, label = 'After high water')
plt.hlines(sample_points_2101_HW.iloc[3].elev_proj, 0, sample_points_2101_HW.iloc[3].dist, color='royalblue', linewidth=2, label = 'High water line')
plt.scatter(sample_points_2101_1.dist, sample_points_2101_1.elev_proj+0.06, color='k', marker='v', linewidth=2, zorder=10, s=100, label = 'Sample location')
# plt.scatter(sample_points_2101_2.dist, sample_points_2101_2.elev_proj+0.04, color='k', marker='v', linewidth=2, zorder=10, s=100)
# plt.scatter(points_2101_eropin.dist, points_2101_eropin.elev_proj+0.06, color='r', marker='v', linewidth=2, zorder=10, s=100)

plt.xlim([0, 75])
plt.ylim([-0.2, 3.5])
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.set_xlabel('Cross-shore distance (m)', fontsize=16)
ax2.set_ylabel('Elevation (m)', fontsize=16)
locs = ['1','3', '2']
for i, txt in enumerate(locs):
    ax2.annotate(txt, (sample_points_2101_1.dist[i]-0.7, sample_points_2101_1.elev_proj[i]+0.28), fontsize=16)

leg = plt.legend(fontsize=16, loc='upper left', framealpha=1)

file_name = 'Noordwijk_profile_210121'
plt.savefig(outputdir + '/CoastalProfiles/' + file_name + '.png')
# plt.close()


#%% Profiles Waldport 23 August 2021
# Load vertical sampling GPS data
filename_2308 = inputdir + '/Waldport/GPS/Waldport_GPS_23082021.txt'
points_2308 = file2df(filename_2308)
selection = [d for d in points_2308.index if 'meas' not in d and 'hwline' not in d and 'eropin' not in d]
points_2308_trsct_all = points_2308.loc[selection]

# Get transect before and after high water
points_2308_trsct1 = points_2308_trsct_all.loc[[d for d in points_2308_trsct_all.index if 'profa' in d]]
points_2308_trsct2 = points_2308_trsct_all.loc[[d for d in points_2308_trsct_all.index if 'profb' in d]]

# create best fit line between the two transects
xnew_2308, ynew_2308, dist_new_2308, intercept_2308, slope_2308 = get_interp_line(points_2308_trsct_all.x, points_2308_trsct_all.y, int(min(points_2308_trsct_all.x)), max(points_2308_trsct_all.x))

# project the measured transects on the best fit line and calculate the interpolated elevation and cross-shore distance
dist_proj_2308_1 = get_dist(points_2308_trsct1.x, points_2308_trsct1.y, intercept_2308, slope_2308, xnew_2308, ynew_2308)
dist_proj_2308_2 = get_dist(points_2308_trsct2.x, points_2308_trsct2.y, intercept_2308, slope_2308, xnew_2308, ynew_2308)

elev_2308_1 = get_interpolation(dist_proj_2308_1, points_2308_trsct1.z, dist_new_2308)
elev_2308_2 = get_interpolation(dist_proj_2308_2, points_2308_trsct2.z, dist_new_2308)

# Get sample locations before HW indicated by measa
sample_points_2308_1 = points_2308.loc[[d for d in points_2308.index if 'measa' in d and 'measa_1' not in d]]
# Get high water line indicated by hw
sample_points_2308_HW = points_2308.loc[[d for d in points_2308.index if 'hw' in d]]
# Get sample locations after HW indicated by measb
sample_points_2308_2 = points_2308.loc[[d for d in points_2308.index if 'measb' in d]]

# Calculate distance and elevation along best fit line for sample locations
sample_points_2308_1['dist'] = get_dist(sample_points_2308_1.x, sample_points_2308_1.y, intercept_2308, slope_2308, xnew_2308, ynew_2308)
sample_points_2308_2['dist'] = get_dist(sample_points_2308_2.x, sample_points_2308_2.y, intercept_2308, slope_2308, xnew_2308, ynew_2308)
sample_points_2308_HW['dist'] = get_dist(sample_points_2308_HW.x, sample_points_2308_HW.y, intercept_2308, slope_2308, xnew_2308, ynew_2308)

sample_points_2308_1['elev_proj'] = get_interpolation(dist_proj_2308_1, points_2308_trsct1.z.values, sample_points_2308_1['dist'])
sample_points_2308_2['elev_proj'] = get_interpolation(dist_proj_2308_2, points_2308_trsct2.z.values, sample_points_2308_2['dist'])
sample_points_2308_HW['elev_proj'] = get_interpolation(dist_proj_2308_2, points_2308_trsct2.z.values, sample_points_2308_HW['dist'])

# Plot profile
fig3, ax3 = plt.subplots(figsize=(11, 3.5))
fig3.subplots_adjust(bottom=0.2)

plt.plot(dist_new_2308, elev_2308_1, 'k', linewidth = 2.5, label = 'Before high water')
plt.plot(dist_new_2308, elev_2308_2, 'grey', linewidth = 2.5, label = 'After high water')
plt.hlines(sample_points_2308_HW.elev_proj, 0, sample_points_2308_HW.dist, color='royalblue', linewidth=2, label = 'High water line')
plt.scatter(sample_points_2308_1.dist, sample_points_2308_1.elev_proj+0.1, color='k', marker='v', linewidth=2, zorder=10, s=100, label='Sample location')
# plt.scatter(sample_points_2308_2.dist, sample_points_2308_2.elev_proj+0.04, color='k', marker='v', linewidth=2, zorder=10, s=100)

plt.xlim([0, 150])
plt.ylim([1, 4])
ax3.tick_params(axis='both', which='major', labelsize=16)
ax3.set_xlabel('Cross-shore distance (m)', fontsize=16)
ax3.set_ylabel('Elevation (m)', fontsize=16)
locs = ['1', '2','3']
for i, txt in enumerate(locs):
    ax3.annotate(txt, (sample_points_2308_1.dist[i]-1.2, sample_points_2308_1.elev_proj[i]+0.28), fontsize=16)

leg = plt.legend(fontsize=16, loc='upper left', framealpha = 1)

file_name = 'Waldport_profile_230821'
plt.savefig(outputdir + '/CoastalProfiles/' + file_name + '.png')

#%% DUCK 1, 2, 3, September 2021
data_dir = inputdir + "Duck/GPS_Laserscans/"

# Load sample locations
points_locs = np.loadtxt(data_dir + 'Duck_GPS_02092021.txt')

# Load netcdf with laserscans
file_3108_2330 = Dataset(data_dir + "FRF-geomorphology_DEMs_duneLidarDEM_20210831_233019.nc")
file_0109_1030 = Dataset(data_dir + "FRF-geomorphology_DEMs_duneLidarDEM_20210901_103017.nc")
file_0109_1330 = Dataset(data_dir + "FRF-geomorphology_DEMs_duneLidarDEM_20210901_133019.nc")
file_0209_0530 = Dataset(data_dir + "FRF-geomorphology_DEMs_duneLidarDEM_20210902_053016.nc")
file_0209_2030 = Dataset(data_dir + "FRF-geomorphology_DEMs_duneLidarDEM_20210902_203018.nc")
file_0309_0930 = Dataset(data_dir + "FRF-geomorphology_DEMs_duneLidarDEM_20210903_093017.nc")

# location of transect
y = 904
y_i = 204

# Get variables
xFRF = file_0109_1030.variables['xFRF'][:]
yFRF = file_0109_1030.variables['yFRF'][:]

# get profile on sampling array
profile_3108_2330 = file_3108_2330.variables['elevation'][:,:][0,y_i,:]
profile_0109_1030 = file_0109_1030.variables['elevation'][:,:][0,y_i,:]
profile_0109_1330 = file_0109_1330.variables['elevation'][:,:][0,y_i,:]
profile_0209_0530 = file_0209_0530.variables['elevation'][:,:][0,y_i,:]
profile_0209_2030 = file_0209_2030.variables['elevation'][:,:][0,y_i,:]
profile_0309_0930 = file_0309_0930.variables['elevation'][:,:][0,y_i,:]

#%% Plot of overview of sample locations
fig4, ax4 = plt.subplots()
ax4.scatter(points_locs[:,2], points_locs[:,1])
for i, txt in enumerate(points_locs[:,0]):
    ax4.annotate(txt, (points_locs[i,2], points_locs[i,1]))

# Get relevant sub section of locations
x_locs, y_locs, elev_locs = points_locs[6:,1], points_locs[6:,2], points_locs[6:,3]

#%% Plot aeolian transport event profiles
fig5, ax5 = plt.subplots(figsize=(11, 3.5))
fig5.subplots_adjust(bottom=0.2)

ax5.plot(xFRF, profile_0109_1030, 'k', linewidth = 2.5, label = 'Before aeolian transport') # 7.30 am 1 September, but this file was missing. However,
    # no change was detected between profile_3108_2330 and profile_0109_1030, so profile_0109_1030 was used as representation for the situation before hw
ax5.plot(xFRF, profile_0109_1330, 'grey', linewidth = 2.5, label = 'During aeolian transport') # 1.30 pm 1 September
ax5.plot(xFRF, profile_0209_0530, 'lightgrey', linestyle = 'dashed', linewidth = 2.5, label = 'After aeolian transport') # 5 am 2 September
ax5.scatter(x_locs[0:3], elev_locs[0:3]+0.08, color='k', marker='v', linewidth=2, zorder=10, s=100)

for i, txt in enumerate(locs):
    ax5.annotate(txt, (x_locs[0:3][i]+0.13, elev_locs[0:3][i]+0.19), fontsize=16)

plt.xlim([67.5, 82.5])
plt.ylim([1, 3])
ax5.tick_params(axis='both', which='major', labelsize=16)
ax5.set_xlabel('Cross-shore distance (m)', fontsize=16)
ax5.set_ylabel('Elevation (m)', fontsize=16)
leg = plt.legend(fontsize=16, loc='lower right', framealpha = 1)

plt.gca().invert_xaxis()   

file_name = 'Duck_profile_aeolian'
plt.savefig(outputdir + '/CoastalProfiles/' + file_name + '.png')

#%% Plot sample locations of detailed sampling
locs_names2 = np.append(points_locs[6:,0], [11, 10])

maxx = max(x_locs)
minz = min(elev_locs)
dist_locs2 = np.append(x_locs, [maxx+1.5, maxx+3])
elev_locs2 = np.append(elev_locs, [minz-0.1, minz-0.2])

fig6, ax6 = plt.subplots(figsize=(18, 3.5))

fig6.subplots_adjust(bottom=0.2)

ax6.scatter(dist_locs2, [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2], color='k', marker='v', linewidth=2, zorder=10, s=100)
for i, txt in enumerate(locs_names2):
    if txt == 7 or txt == 8 or txt == 9:
        ax6.annotate(int(txt-6), (dist_locs2[i]+0.1, 0.32), fontsize = 18)
    else:
        ax6.annotate(int(txt), (dist_locs2[i]+0.1, 0.32), fontsize = 18)
        
ax6.tick_params(bottom=True, labelbottom=True, left=False, labelleft=False)
ax6.tick_params(labelsize=18) 
ax6.set_xlabel('Cross-shore distance (m)', fontsize=20)

plt.ylim([0, 0.4])
# plt.xlim([69.5, 83])
plt.gca().invert_xaxis()    
plt.show()

# leg = plt.legend(fontsize=16, loc='lower right')

file_name = 'Duck_profile_detailedsampling_locs'
plt.savefig(outputdir + '/CoastalProfiles/' + file_name + '.png')

#%% Plot coastal profile of detailed sampling
fig7, ax7 = plt.subplots(figsize=(18, 3.5))

fig7.subplots_adjust(bottom=0.2)

elev_locs2 = [1.808, 2.325 , 2.343, 1.999, 2.291, 2.316, 2.334, 2.341, 1.668, 1.553]

ax7.scatter(dist_locs2, elev_locs2, color='k', marker='v', linewidth=2, zorder=10, s=100)
for i, txt in enumerate(locs_names2):
    if txt == 7 or txt == 8 or txt == 9:
        ax7.annotate(int(txt-6), (dist_locs2[i]+0.08, elev_locs2[i]+0.1), fontsize = 18)
    else:
        ax7.annotate(int(txt), (dist_locs2[i]+0.08, elev_locs2[i]+0.1), fontsize = 18)

ax7.plot(xFRF, profile_0209_0530, 'k', linewidth = 2.5, label = 'After aeolian transport') # Is combo of 5 and 7 am on 2 September 

# ax7.tick_params(bottom=True, labelbottom=True, left=False, labelleft=False)
ax7.tick_params(labelsize=18) 
ax7.set_xlabel('Cross-shore distance (m)', fontsize=20)
ax7.set_ylabel('Elevation (m)', fontsize=20)

plt.ylim([1.3, 2.6])
plt.xlim([69.5, 83])
plt.gca().invert_xaxis()    
plt.show()

# leg = plt.legend(fontsize=16, loc='lower right')

file_name = 'Duck_profile_detailedsampling'
plt.savefig(outputdir + '/CoastalProfiles/' + file_name + '.png')

#%% Plot sampling locations of high water sampling

fig8, ax8 = plt.subplots(figsize=(10, 3))
fig8.subplots_adjust(bottom=0.25)

ero_locs = [x_locs[3], x_locs[0], maxx+1.5, maxx+3]

locs_names3 = ['12', '1', '11', '10']
y_locs = [elev_locs[3], elev_locs[0], elev_locs[0]-0.1, elev_locs[0]-0.2]

ax8.scatter(ero_locs, y_locs, color='k', marker='v', linewidth=2, zorder=10, s=100, label = 'Sample locations')
for i, txt in enumerate(locs_names3):
    ax8.annotate(int(txt), (ero_locs[i]+0.1, y_locs[i]+0.15), fontsize=18)
    # GPS locations are from 6 pm 2 September, had already been influence on scarp! #10 and #11 were lost to the waves, so their location was approximated.

# Note that the laser scans are included in this, so they can be uncommented to show the measured morphological change.
# However, their resolution was too low to get the exact shape of the scarp. Based on these laseer scans and sketches from the field, the coastal profiles were drawn as seen in the final figure.
# ax8.plot(xFRF, profile_0209_0530, 'k', linewidth = 2.5, label = 'Before high water') # Combo of 5 and 7 am on 2 September
# ax8.plot(xFRF, profile_0309_0930, 'k', linewidth = 2.5, label = 'After high water') # 9 am 3 September
    
ax8.tick_params(labelsize=18) 
ax8.set_xlabel('Cross-shore distance (m)', fontsize=18)    
ax8.set_ylabel('Elevation (m)', fontsize=18)    
    
plt.xlim([77, 83])
plt.ylim([1, 2.7])

plt.gca().invert_xaxis()    
plt.show()

# leg = plt.legend(fontsize=16, loc='lower right')

file_name = 'Duck_marine'
plt.savefig(outputdir + '/CoastalProfiles/' + file_name + '.png')


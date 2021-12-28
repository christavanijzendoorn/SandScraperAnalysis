# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 08:48:04 2021

@author: cijzendoornvan
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def calculate_percentages(weights, diameters):
     total_mass = weights.sum()
     percentages = []
     for count, sieve in enumerate(diameters):
         cumulative_mass = sum([weights.values[i] for i in range(count + 1)])
         percentage = ((total_mass - cumulative_mass) / total_mass) * 100
         percentages.append(percentage)
     return percentages

#%%
inputdir = 'C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Data'
outputdir = 'C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Figures/'

#%% Load and plot for WALDPORT

#Put the Directory here with all the files you want to load
directory_WP = inputdir + "/Waldport/Grainsizes/Camsizer_files/"
filename = 'WaldportChrista_B3_14mm.xle'

f = os.path.join(directory_WP, filename)
if os.path.isfile(f):
    print(f)
                
    data = np.loadtxt(f, delimiter='\t', skiprows=31, encoding = "utf-16")
    
    #determine grain size bounds
    lower_bound_size_mm = np.array(data[:,0])
    upper_bound_size_mm = np.array(data[:,1])
    upper_bound_size_mm[-1] = 10
    # grain_size_mm = lower_bound_size_mm/2 + upper_bound_size_mm/2 #if you want to make this more comparable to seiving you should just make this the lower_bound size rather than the middle value
    grain_size_mm = lower_bound_size_mm #if you want to make this more comparable to seiving you should just make this the lower_bound size rather than the middle value
    
    #only include sand size from herein out
    iuse = np.where((grain_size_mm <= 2) & (grain_size_mm >= 0.02))
    grain_size_mm = np.ndarray.flatten(grain_size_mm[iuse])
    
    
    #store other shape data
    p= np.ndarray.flatten(data[iuse,2])
    spht = np.ndarray.flatten(data[iuse,4])
    symm = np.ndarray.flatten(data[iuse,5])
    bl = np.ndarray.flatten(data[iuse,6])
    pdn = np.ndarray.flatten(data[iuse,7])
    q= np.ndarray.flatten(data[iuse,3])

    #fix q to remove the non-sand values
    minq = np.min(q)
    maxq = np.max(q)
    q_corr = np.ndarray.flatten(100*(q-minq)/(maxq-minq))
    
   
    # plot distribution
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    fig1.subplots_adjust(bottom=0.2)
    ax1.plot(grain_size_mm, p, '-', color = 'royalblue', linewidth = 3, label = 'Waldport')
    ax1.set_xlabel("Grain size ($\mu$m)", fontsize = 20)
    ax1.set_ylabel("Percentage (%)", fontsize = 20)
    ax1.set_xlim([0., 1])   
    
#%% Load and plot for NOORDWIJK

# Load Excel sheet with overview of all samples
directory_NW = inputdir + "/Noordwijk/Grainsizes/Sievesheets//"
sieve_files = os.listdir(directory_NW)

samplecode = 'V052_04_2mm'
sample = [f for f in sieve_files if samplecode in f and f[0] != '~']

# Retrieve sieving data (i.e. weight percentages) from sieve sheet
sievedata = pd.read_excel(directory_NW + sample[0], usecols = "A:H", engine='openpyxl')
diameters = [3.35, 2, 1.18, 0.600, 0.425, 0.300, 0.212, 0.150, 0.063, 0] # in mm
weights = sievedata[12:22]['Unnamed: 4']
comments = sievedata[12:22]['Unnamed: 7']
percentages = calculate_percentages(weights, diameters)

q = []
for i, x in enumerate(percentages):
    if i == 0:
        q.append(0)
    else:
        q.append((percentages[i-1] - x) / (200/9))

from scipy import interpolate
f = interpolate.interp1d(diameters, q)
q_intp = f(grain_size_mm)

ax1.plot(diameters, q, 'D-', markersize = 6, color = 'forestgreen', linewidth = 3, label = 'Noordwijk')

#%% Load and plot for DUCK

#Put the Directory here with all the files you want to load
directory_Duck = inputdir + "/Duck/Grainsizes/Camsizer_files/"
filename = 'DunexChrista_sample2_14mm.xle'

f = os.path.join(directory_Duck, filename)
if os.path.isfile(f):
    print(f)
                
    data = np.loadtxt(f, delimiter='\t', skiprows=31, encoding = "utf-16")
    
    #determine grain size bounds
    lower_bound_size_mm = np.array(data[:,0])
    upper_bound_size_mm = np.array(data[:,1])
    upper_bound_size_mm[-1] = 10
    # grain_size_mm = lower_bound_size_mm/2 + upper_bound_size_mm/2 
    grain_size_mm = lower_bound_size_mm #if you want to make this more comparable to seiving you should just make this the lower_bound size rather than the middle value
    
    #only include sand size from herein out
    iuse = np.where((grain_size_mm <= 2) & (grain_size_mm >= 0.02))
    grain_size_mm = np.ndarray.flatten(grain_size_mm[iuse])
    
    
    #store other shape data
    p= np.ndarray.flatten(data[iuse,2])
    spht = np.ndarray.flatten(data[iuse,4])
    symm = np.ndarray.flatten(data[iuse,5])
    bl = np.ndarray.flatten(data[iuse,6])
    pdn = np.ndarray.flatten(data[iuse,7])
    q= np.ndarray.flatten(data[iuse,3])

    #fix q to remove the non-sand values
    minq = np.min(q)
    maxq = np.max(q)
    q_corr = np.ndarray.flatten(100*(q-minq)/(maxq-minq))
  
    ax1.plot(grain_size_mm, p, '-', color = 'goldenrod', linewidth = 3, label = 'Duck') 
   
# fig1.tight_layout()
ax1.legend(fontsize = 20)
ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_xticklabels(['0', '200', '400', '600', '800', '1000'])
ax1.tick_params(axis='both', which='major', labelsize=18)
plt.show()

file_name = 'Figure2c_grainsizedistribution'
plt.savefig(outputdir + file_name + '.png')

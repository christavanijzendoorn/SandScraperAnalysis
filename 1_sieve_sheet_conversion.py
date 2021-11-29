
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:16:01 2020

@author: cijzendoornvan
"""

#################
#### FUNCTIONS

def get_filtered_dvalues(comment_filter, weights, diameters):
    weights_filtered = comment_filter*weights

    total_weight_filtered = sum(weights_filtered)
    percentages_retained = 100*(weights_filtered/total_weight_filtered)
    percentages_filtered = []
    
    i = 0
    while i < 10:
        if i == 0:
            percentages_filtered.append(100 - percentages_retained[i+12])
        else:
            percentages_filtered.append(percentages_filtered[i-1] - percentages_retained[i+12])
        i += 1
        
    f_linear = interp1d(list(percentages_filtered), diameters, fill_value='extrapolate')
    
    d10_lin = f_linear(10)
    d16_lin = f_linear(16)
    d25_lin = f_linear(25)
    d50_lin = f_linear(50)
    d75_lin = f_linear(75)
    d84_lin = f_linear(84)
    d90_lin = f_linear(90)
    
    return d10_lin, d16_lin, d25_lin, d50_lin, d75_lin, d84_lin, d90_lin

####################
#### INITIALISATION

import os
import pandas as pd
from scipy.interpolate import interp1d

inputdir = 'C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Data/'

# Load Excel sheet with overview of all samples
Dir2SampleData = inputdir + "/Noordwijk/Grainsizes//"
SampleData = "Data_Noordwijk.xlsx"
sampledata = pd.read_excel(Dir2SampleData + SampleData, usecols = "A:P", engine='openpyxl')
sampledata = sampledata.set_index(['Type', 'Date', 'Location', 'Depth'])

# Load all sieve sheet names as present in Siev_sheets folder
Dir2SieveSheets = Dir2SampleData + "Sievesheets//"
sieve_files = os.listdir(Dir2SieveSheets)

####################
#### GET grain sizes

# For all available sieve sheets, fill in the D10, D16, D25, D50, D75, D84 and D90 into the overview file
for sample in sieve_files:
    samplecode = sample[0] # Get sample code: D, N, S, C, V

    # Retrieve sieving data (i.e. weight percentages) from sieve sheet
    sievedata = pd.read_excel(Dir2SieveSheets + sample, usecols = "A:H", engine='openpyxl')
    diameters = [3.35, 2, 1.18, 0.600, 0.425, 0.300, 0.212, 0.150, 0.063, 0] # in mm
    weights = sievedata[12:22]['Unnamed: 4']
    weight_total = sievedata['Unnamed: 4'][22]
    comments = sievedata[12:22]['Unnamed: 7']
    percentages = sievedata[12:22]['Unnamed: 6']
    
    f_linear = interp1d(list(percentages), diameters, fill_value='extrapolate')
    
    # Retrieve D10, D50 and D90 based on interpolation
    d10_lin = f_linear(10)
    d16_lin = f_linear(16)
    d25_lin = f_linear(25)
    d50_lin = f_linear(50)
    d75_lin = f_linear(75)
    d84_lin = f_linear(84)
    d90_lin = f_linear(90)
    
    # Get sample info from sievesheet file name
    sample_info = sample.split('_')
    sampledate = int(sample_info[0][1:]) # Get sample date, format: DDM    
    sampleno = '_' + sample_info[1]
    
    sampledepth = sample_info[2].split('.')[0] # Get sample depth
    if len(sampledepth) == 3:
        sampledepth = sampledepth[0] + ' ' + sampledepth[1:]
    else:
        sampledepth = sampledepth[0:2] + ' ' + sampledepth[2:]
        
    # Save values in overview of samples
    sampledata.loc[(samplecode, sampledate, sampleno, sampledepth),'D10'] = d10_lin
    sampledata.loc[(samplecode, sampledate, sampleno, sampledepth),'D16'] = d16_lin
    sampledata.loc[(samplecode, sampledate, sampleno, sampledepth),'D25'] = d25_lin
    sampledata.loc[(samplecode, sampledate, sampleno, sampledepth),'D50'] = d50_lin
    sampledata.loc[(samplecode, sampledate, sampleno, sampledepth),'D75'] = d75_lin
    sampledata.loc[(samplecode, sampledate, sampleno, sampledepth),'D84'] = d84_lin
    sampledata.loc[(samplecode, sampledate, sampleno, sampledepth),'D90'] = d90_lin      

# sampledata['D50'].mean()
        
# Separately save updated sample data as excel sheet
SampleData_D = "Grainsize_data_Noordwijk.xlsx"
sampledata.to_excel(Dir2SampleData + SampleData_D)

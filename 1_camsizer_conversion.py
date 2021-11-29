#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:04:45 2021

@author: rdchlntc
"""

import numpy as np
import os
import xlsxwriter

#Put the Directory here with all the files you want to load
inputdir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Data/"
# camsizer_dir = "/Duck/Grainsizes/Camsizer_files/"
# camsizer_dir = "/Duck/Grainsizes/Camsizer_files/RepetitionSamples/"
camsizer_dir = "/Waldport/Grainsizes/Camsizer_files/"
directory = inputdir + camsizer_dir

#get corresponding outputdir
if 'Duck' in camsizer_dir and 'Repetition' in camsizer_dir:
    outputfile = inputdir + '/Duck/Grainsizes/Grainsize_data_Duck_repetition.xlsx'
elif 'Duck' in camsizer_dir and 'Repetition' not in camsizer_dir:
    outputfile = inputdir + '/Duck/Grainsizes/Grainsize_data_Duck.xlsx'
else:
    outputfile = inputdir + '/Waldport/Grainsizes/Grainsize_data_Waldport.xlsx'

#first loop through to determine how many xle files are present
os.chdir(directory)
count = -1
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f) and filename.endswith('.xle'):
        count = count+1
        
#set up corresponding variables for storage + output
d2 = np.zeros([count+1])  
d5 = np.zeros([count+1])  
d10 = np.zeros([count+1])  
d16 = np.zeros([count+1])  
d25 = np.zeros([count+1])  
d50 = np.zeros([count+1])  
d75 = np.zeros([count+1])  
d84 = np.zeros([count+1])  
d90 = np.zeros([count+1])  
d95 = np.zeros([count+1])  
d98 = np.zeros([count+1])  
bl_avg = np.zeros([count+1])  
spht_avg = np.zeros([count+1])  
symm_avg = np.zeros([count+1])  
perc_total = np.zeros([count+1])  
perc_very_coarse_sand = np.zeros([count+1])  
perc_coarse_sand = np.zeros([count+1])  
perc_medium_sand = np.zeros([count+1])  
perc_fine_sand = np.zeros([count+1])  
perc_very_fine_sand = np.zeros([count+1])  

#set up excel spreadsheet
workbook = xlsxwriter.Workbook(outputfile)
worksheet = workbook.add_worksheet()
worksheet.write(0,0, 'Filename')
worksheet.write(0,1, 'D2(mm)')
worksheet.write(0,2, 'D5(mm)')
worksheet.write(0,3, 'D10(mm)')
worksheet.write(0,4, 'D16(mm)')
worksheet.write(0,5, 'D25(mm)')
worksheet.write(0,6, 'D50(mm)')
worksheet.write(0,7, 'D75(mm)')
worksheet.write(0,8, 'D84(mm)')
worksheet.write(0,9, 'D90(mm)')
worksheet.write(0,10, 'D95(mm)')
worksheet.write(0,11, 'D98(mm)')
worksheet.write(0,12, 'Symmetry_Avg')
worksheet.write(0,13, 'Sphericity_Avg')
worksheet.write(0,14, 'B/L_Avg')
worksheet.write(0,15, 'Perc_VeryCoarseSand')
worksheet.write(0,16, 'Perc_CoarseSand')
worksheet.write(0,17, 'Perc_MediumSand')
worksheet.write(0,18, 'Perc_FineSand')
worksheet.write(0,19, 'Perc_VeryFineSand')

#loop through again to actually calculate data
count = -1
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f) and filename.endswith('.xle'):
        print(f)
                
        data = np.loadtxt(filename, delimiter='\t', skiprows=31, encoding = "utf-16")
        count = count + 1
        
        #determine grain size bounds
        lower_bound_size_mm = np.array(data[:,0])
        upper_bound_size_mm = np.array(data[:,1])
        upper_bound_size_mm[-1] = 10
        grain_size_mm = lower_bound_size_mm/2 + upper_bound_size_mm/2 #if you want to make this more comparable to seiving you should just make this the lower_bound size rather than the middle value
        
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
             
        #calc relevant stats
        d2[count] = np.interp(2, q_corr, grain_size_mm)
        d5[count] = np.interp(5, q_corr, grain_size_mm)
        d10[count] = np.interp(10, q_corr, grain_size_mm)
        d16[count] = np.interp(16, q_corr, grain_size_mm)
        d25[count] = np.interp(25, q_corr, grain_size_mm)
        d50[count] = np.interp(50, q_corr, grain_size_mm)
        d75[count] = np.interp(75, q_corr, grain_size_mm)
        d84[count] = np.interp(84, q_corr, grain_size_mm)
        d90[count] = np.interp(90, q_corr, grain_size_mm)
        d95[count] = np.interp(95, q_corr, grain_size_mm)
        d98[count] = np.interp(98, q_corr, grain_size_mm)       
        ifind = np.where((d25[count] <= grain_size_mm) & (grain_size_mm <= d75[count]))
        symm_avg[count] = np.nanmean(symm[ifind])        
        spht_avg[count] = np.nanmean(spht[ifind])        
        bl_avg[count] = np.nanmean(bl[ifind])   
                
        
        #ifind = np.where((grain_size_mm >= 1) & (grain_size_mm < 2))
        perc_total[count] = np.sum(p[np.where((grain_size_mm >= 0.0625) & (grain_size_mm < 2))])
        perc_very_coarse_sand[count] = np.sum(p[np.where((grain_size_mm >= 1) & (grain_size_mm < 2))])/perc_total[count] * 100
        perc_coarse_sand[count] = np.sum(p[np.where((grain_size_mm >= 0.5) & (grain_size_mm < 1))])/perc_total[count] * 100
        perc_medium_sand[count] = np.sum(p[np.where((grain_size_mm >= 0.25) & (grain_size_mm < 0.5))])/perc_total[count] * 100
        perc_fine_sand[count] = np.sum(p[np.where((grain_size_mm >= 0.125) & (grain_size_mm < 0.25))])/perc_total[count] * 100
        perc_very_fine_sand[count] = np.sum(p[np.where((grain_size_mm >= 0.0625) & (grain_size_mm < 0.125))])/perc_total[count] * 100
        
        
        #write out data                
        worksheet.write(count+1,0, filename)     
        worksheet.write(count+1,1, d2[count])
        worksheet.write(count+1,2, d5[count])
        worksheet.write(count+1,3, d10[count])
        worksheet.write(count+1,4, d16[count])
        worksheet.write(count+1,5, d25[count])
        worksheet.write(count+1,6, d50[count])
        worksheet.write(count+1,7, d75[count])
        worksheet.write(count+1,8, d84[count])
        worksheet.write(count+1,9, d90[count])
        worksheet.write(count+1,10, d95[count])
        worksheet.write(count+1,11, d98[count])
        worksheet.write(count+1,12, symm_avg[count])
        worksheet.write(count+1,13, spht_avg[count])
        worksheet.write(count+1,14, bl_avg[count])
        worksheet.write(count+1,15, perc_very_coarse_sand[count])
        worksheet.write(count+1,16, perc_coarse_sand[count])
        worksheet.write(count+1,17, perc_medium_sand[count])
        worksheet.write(count+1,18, perc_fine_sand[count])
        worksheet.write(count+1,19, perc_very_fine_sand[count])
                                
#close and save excel spreadsheet        
workbook.close()
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 20:00:08 2021

@author: cijzendoornvan
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

camsizer_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Data/Noordwijk/Grainsizes/"
input_file = 'Grainsize_data_Noordwijk.xlsx'
output_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Figures/VerticalSampling/crosssections_NW/"

data = pd.read_excel(camsizer_dir + input_file, engine='openpyxl')

depths = [2, 4, 8, 14, 20, 26, 32, 38, 44, 50]

# samples = ['A2', 'A3', 'A4', 'B2', 'B3', 'B4', 'C4', 'D4']


#%% Create all grain size cross sections

data['Date'].fillna(method='ffill', inplace=True)
data['Type'].fillna(method='ffill', inplace=True)
data['Location'].fillna(method='ffill', inplace=True)
data = data.set_index(['Type', 'Date', 'Location'])

sample_code = 'V'   
dates = data.index.get_level_values('Date').unique()

for date in dates:
    data_date = data.xs(date, level=1, drop_level=False)    
    locations = data_date.index.get_level_values('Location').unique()

    for i, loc in enumerate(locations):
        fig = plt.figure(figsize=(4,8))
        ax = fig.add_axes([0.15, 0.1, 0.5, 0.70])
        data_loc = data_date.xs(loc, level=2, drop_level=False)
        
        norm  = colors.Normalize(vmin=200, vmax=400)
        scalarMap = cmx.ScalarMappable(norm=norm, cmap='hot_r')
        
        for j, layer in enumerate(data_loc['Depth']):
        
            if j == 0:
                upper_boundary = 0
            else: 
                upper_boundary = -1 * int(data_loc['Depth'][j-1][0:2])
            lower_boundary = -1 * int(layer[0:2])
            
            colorVal = scalarMap.to_rgba(data_loc['D50'][j]*1000)
            ax.fill_between([0.1, 1.], [upper_boundary, upper_boundary],
                                 [lower_boundary, lower_boundary],
                                 facecolor=colorVal, edgecolor = 'k')
            
        if int(data_loc['Depth'][-1][0:2]) < 50:
            ax.fill_between([0.1, 1.], [-1* int(data_loc['Depth'][-1][0:2]), -1* int(data_loc['Depth'][-1][0:2])],
                     [-50, -50], facecolor='white') # , edgecolor = 'k'
            
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False, left=True, labelleft=True)
        # ax.annotate(gdf_set_loc['time_str'][j], (0.3,2), fontsize = 20)
        # ax.text(0.30, -52.5, 'std dev: '  + '{:.3f}'.format(gdf_set_loc['D50'].std()), fontsize = 16)
        # ax.text(0.30, -51.2, 'mean: ' + '{:.3f}'.format(gdf_set_loc['D50'].mean()), fontsize = 16)
        # ax.text(0.30, 2, sample_code + str(int(date)) + loc, fontsize = 20)
        # ax.text(0.05, -22.8, 'min: ' + '{:.3f}'.format(gdf_set_loc['D50'].min()), fontsize = 10)
        # ax.text(0.05, -23.7, 'max: ' + '{:.3f}'.format(gdf_set_loc['D50'].max()), fontsize = 10)
        ax.tick_params(labelsize=20) 
        
        # Only show ticks on the left and bottom spines
        if i == 0:
            ax.set_ylabel('depth (mm)', fontsize = 20)
            ax.yaxis.set_ticks_position('left')
            ax.set_yticks([0, -2, -4, -8, -14, -20])
    
           
        cax = fig.add_axes([0.7, 0.13, 0.04, 0.64])
        fig.colorbar(scalarMap, cax=cax, orientation='vertical')
        cax.tick_params(labelsize=20) 
        # cax.set_ylabel('Grain size - D50 ($\mu$m)', fontsize = 20) 
        
        file_name = sample_code + str(int(date)) + loc
        plt.savefig(output_dir + file_name + '_plot.png')
        plt.close()
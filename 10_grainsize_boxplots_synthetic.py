# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 20:28:57 2021

@author: cijzendoornvan
"""

import numpy as np
import math 

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (int(average), round(math.sqrt(variance), 0))

thicknesses = [2, 2, 4, 6, 6, 6, 6, 6, 6, 6]
#%%#### PLOT VERTICAL SAMPLING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

grainsizes_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Data/"
grainsizes_file = 'Grainsize_data_synthetic.xlsx'
output_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Figures/VerticalSampling/"

data = pd.read_excel(grainsizes_dir + grainsizes_file, engine='openpyxl')

data['Type'].fillna(method='ffill', inplace=True)
data['Location'].fillna(method='ffill', inplace=True)
data = data.set_index(['Type', 'Location'])

data['Depth'] = [int(d.replace(' mm', '')) for d in data['Depth']]

sample_code = 'V'   
locations = ['_01', '_02', '_03']

# Prepare coloring based on median value
norm  = colors.Normalize(vmin=data['D50'].min()*1000, vmax=data['D50'].max()*1000)
scalarMap = cmx.ScalarMappable(norm=norm, cmap='hot_r')
data['colorval'] = data['D50']
data['colorval'] = [scalarMap.to_rgba(d*1000) for d in data['colorval']]
#%%
depths = [2, 4, 8, 14, 20, 26, 32, 38, 44, 50]

locations = ['_01', '_02', '_03']
cols = ['1', '2', '3']
plot_loc_col = [0, 1, 2]

fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12, 4))
fig.subplots_adjust(right=0.85)
fig.subplots_adjust(left=0.14)
fig.subplots_adjust(bottom=0.17)
fig.subplots_adjust(top=0.9)

for i, s in enumerate(locations):
    print(s)
    
    # Select data belonging to sample
    location = data.xs(s, level=1, drop_level=False)
    location_sort = location.set_index('Depth').sort_index() # set index to depth
    location_plot = location_sort[['D16', 'D25', 'D50', 'D75', 'D84']].values.T * 1000 # convert to values in mm's
    
    colors = location_sort['colorval']    
    
    # include vertical grid
    axs[plot_loc_col[i]].xaxis.grid(True) 
    
    # Add average and standard deviation of median grain sizes
    avg, stddev = weighted_avg_and_std(location_sort.loc[:, 'D50']*1000, thicknesses[:len(location_sort.loc[:, 'D50'])])
    props = dict(facecolor='white', edgecolor = 'white')
    axs[plot_loc_col[i]].axvline(x=avg, color = 'dimgrey', linewidth=2)
    #axs[plot_loc_col[i]].axhline(y=0.17, color = 'grey', linewidth=0.5)
    axs[plot_loc_col[i]].text(avg-150, -0.3,'$\u00f8_{50}$ = ' + str(avg), fontsize=14, bbox = props) #+ '\t  $\u03C3_{50}$ = ' + str(stddev)
        
    # Plot boxplot in subplot
    bplot = axs[plot_loc_col[i]].boxplot(location_plot, vert=False, showfliers=False, whis=50.0, widths=0.5, patch_artist = True, boxprops=dict(color='black'), medianprops=dict(linewidth=1.5, color='grey'))
                                                  # boxprops=dict(facecolor=c, color=c),
                                                  # capprops=dict(color=c),
                                                  # whiskerprops=dict(color=c),
                                                  # flierprops=dict(color=c, markeredgecolor=c),
                                                  # )
    # Set limits and ticks of subplot    
    axs[plot_loc_col[i]].set_xlim(200, 550)
    axs[plot_loc_col[i]].set_ylim(-1.2, 10.5)
    axs[plot_loc_col[i]].set_yticks(range(1,11))
    axs[plot_loc_col[i]].set_yticklabels(['2', '4', '8', '14', '20', '26', '32', '38', '44', '50'])
    
    axs[plot_loc_col[i]].tick_params(axis='both', which='major', labelsize=16)
    axs[plot_loc_col[i]].tick_params(axis='both', which='minor', labelsize=16)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
# Flip axes so depth is shown correctly    
plt.gca().invert_yaxis()    
# plt.rcParams.update({'axes.titlesize': 'large'})

# Set axes labels and only show them for outer axes 
# plt.setp(axs.flat, xlabel='Grain size ($\mu$m)', ylabel='Depth (mm)')    
for ax in axs.flat:
    ax.set_xlabel('Grain size ($\mu$m)', fontsize=18)
    ax.set_ylabel('Depth (mm)', fontsize=18)
    ax.label_outer()

# Add colorbar    
cax = fig.add_axes([0.87, 0.10, 0.02, 0.80])
fig.colorbar(scalarMap, cax=cax, orientation='vertical')
cax.tick_params(labelsize=18) 
cax.set_ylabel('Median grain size ($\mu$m)', fontsize = 18) 
      
# Finish layout
plt.show()

# Save as file
file_name = 'Grain_size_vertical'
plt.savefig(output_dir + file_name + '_synthetic_boxplot.png', dpi=800)
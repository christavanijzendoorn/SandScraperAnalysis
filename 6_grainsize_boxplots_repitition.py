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
    return (int(average), int(round(math.sqrt(variance), 0)))

thicknesses = [2, 2, 4, 6, 6, 6, 6, 6, 6, 6]

#%%#### PLOT VERTICAL SAMPLING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

vmin = 240
vmax = 390

grainsizes_dir_NW = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Data/Noordwijk/Grainsizes/"
grainsizes_file_NW = 'Grainsize_data_Noordwijk.xlsx'
output_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Figures/VerticalSampling/"

data_NW = pd.read_excel(grainsizes_dir_NW + grainsizes_file_NW, engine='openpyxl')

data_NW['Date'].fillna(method='ffill', inplace=True)
data_NW['Type'].fillna(method='ffill', inplace=True)
data_NW['Location'].fillna(method='ffill', inplace=True)
data_NW = data_NW.set_index(['Type', 'Date', 'Location'])

data_NW['Depth'] = [int(d.replace(' mm', '')) for d in data_NW['Depth']]

sample_code = 'V'   
date = 2111
date_str = '21 November'
locations = ['_05', '_06']

data_NW = data_NW.xs(date, level=1, drop_level=False)

# Prepare coloring based on median value
norm  = colors.Normalize(vmin=vmin, vmax=vmax)
scalarMap = cmx.ScalarMappable(norm=norm, cmap='hot_r')
data_NW['colorval'] = data_NW['D50']
data_NW['colorval'] = [scalarMap.to_rgba(d*1000) for d in data_NW['colorval']]
#%%#### PLOT VERTICAL SAMPLING

camsizer_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Data/Duck/Grainsizes/"
input_file = 'Grainsize_data_Duck_repetition.xlsx'

data_Duck = pd.read_excel(camsizer_dir + input_file, engine='openpyxl')

depths = []
for i, d in enumerate(data_Duck['Filename'].values):
    if '_2.'in d:
        depth = int(d.split('_')[-2].replace('mm',''))
    else:        
        depth = int(d.split('_')[-1].split('.')[0].replace('mm',''))
    depths.append(depth)
data_Duck['Depth'] = depths

# Select data belonging to sample
Duck = data_Duck.loc[data_Duck['Filename'].str.contains('sample2')]

# Prepare coloring based on median value
norm  = colors.Normalize(vmin=vmin, vmax=vmax)
scalarMap = cmx.ScalarMappable(norm=norm, cmap='hot_r')
Duck['colorval'] = Duck['D50(mm)']
Duck['colorval'] = [scalarMap.to_rgba(d*1000) for d in Duck['colorval']]

#%%
depths = [2, 4, 8, 14, 20, 26, 32, 38, 44, 50]

locations = ['_05', '_06', 'sample2_', 'sample2rep_']
cols = ['Noordwijk 1', 'Noordwijk 2', 'Duck 1', 'Duck 2']
plot_loc_col = [0, 1, 2, 3]

fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(10, 4))
fig.subplots_adjust(right=0.83)
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(top=0.8)

for i, s in enumerate(locations):
    print(s)
    
    if 'sample' in s:
        # Select data belonging to sample
        location = Duck.loc[Duck['Filename'].str.contains(s)]
        location_sort = location.set_index('Depth').sort_index() # set index to depth
        location_plot = location_sort[['D16(mm)', 'D25(mm)', 'D50(mm)', 'D75(mm)', 'D84(mm)']].values.T * 1000 
        
        # Add average and standard deviation of median grain sizes
        avg, stddev = weighted_avg_and_std(location_sort.loc[:, 'D50(mm)']*1000, thicknesses[:len(location_sort.loc[:, 'D50(mm)'])])
        props = dict(facecolor='white', edgecolor = 'white')
        axs[plot_loc_col[i]].axhline(y=10.65, color = 'grey', linewidth=0.5)
        axs[plot_loc_col[i]].text(175, 11.95,'$\u00f8_{50}$ = ' + str(avg) , fontsize=15, bbox = props)
            

    if '_0' in s:
        # Select data belonging to sample
        location = data_NW.xs(s, level=2, drop_level=False)
        location_sort = location.set_index('Depth').sort_index() # set index to depth
        location_plot = location_sort[['D16', 'D25', 'D50', 'D75', 'D84']].values.T * 1000 # convert to values in mm's
        
        # Add average and standard deviation of median grain sizes
        avg, stddev = weighted_avg_and_std(location_sort.loc[:, 'D50']*1000, thicknesses[:len(location_sort.loc[:, 'D50'])])
        props = dict(facecolor='white', edgecolor = 'white')
        axs[plot_loc_col[i]].axhline(y=10.65, color = 'grey', linewidth=0.5)
        axs[plot_loc_col[i]].text(175, 11.95,'$\u00f8_{50}$ = ' + str(avg), fontsize=15, bbox = props)        
    
    colors = location_sort['colorval']    
    
    # include vertical grid
    axs[plot_loc_col[i]].xaxis.grid(True) 
    
    # Plot boxplot in subplot
    bplot = axs[plot_loc_col[i]].boxplot(location_plot, vert=False, showfliers=False, whis=5.0, widths=0.5, patch_artist = True, boxprops=dict(color='black'), medianprops=dict(linewidth=2.0, color='grey'))
                                                  # boxprops=dict(facecolor=c, color=c),
                                                  # capprops=dict(color=c),
                                                  # whiskerprops=dict(color=c),
                                                  # flierprops=dict(color=c, markeredgecolor=c),
                                                  # )

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    # Set limits and ticks of subplot    
    axs[plot_loc_col[i]].set_xlim(150, 550)
    axs[plot_loc_col[i]].set_ylim(0, 12.6)
    axs[plot_loc_col[i]].set_yticks(range(1,11))
    axs[plot_loc_col[i]].set_yticklabels(depths)
    
    axs[plot_loc_col[i]].tick_params(axis='both', which='major', labelsize=16)
    axs[plot_loc_col[i]].tick_params(axis='both', which='minor', labelsize=16)    
    
# Flip axes so depth is shown correctly    
plt.gca().invert_yaxis()    
plt.rcParams.update({'axes.titlesize': 'large'})

# Set axes labels and only show them for outer axes 
# plt.setp(axs.flat, xlabel='Grain size ($\mu$m)', ylabel='Depth (mm)')    
for ax in axs.flat:
    # ax.set_xlabel('Grain size ($\mu$m)', fontsize=18)
    ax.set_ylabel('Depth (mm)', fontsize=18)
    ax.label_outer()
    
# Add x-label and subplot titles    
pad = 5
ax.annotate('Noordwijk', xy=(-200, 1.5), xytext=(-230, 120), 
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='xx-large', ha='right', va='center')    

ax.annotate('Grain size ($\mu$m)', xy=(-200, 1.5), xytext=(-200, -125), 
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='xx-large', ha='right', va='center')   

pad = 5
ax.annotate('Duck', xy=(-200, 1.5), xytext=(15, 120), 
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='xx-large', ha='right', va='center')    

ax.annotate('Grain size ($\mu$m)', xy=(-200, 1.5), xytext=(65, -125), 
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='xx-large', ha='right', va='center')   

# Add colorbar    
cax = fig.add_axes([0.85, 0.20, 0.02, 0.60])
fig.colorbar(scalarMap, cax=cax, orientation='vertical')
cax.tick_params(labelsize=18) 
cax.set_ylabel('Median grain size ($\mu$m)', fontsize = 18) 
      
# Finish layout
plt.show()

# Save as file
file_name = 'Grain_size_vertical'
plt.savefig(output_dir + file_name + '_repetition_boxplot.png')

#%% Calculate averages
thicknesses = [2, 2, 4, 6, 6, 6, 6, 6, 6, 6]

NW_1 = data_NW.xs(locations[0], level=2, drop_level=False)
NW_2 = data_NW.xs(locations[1], level=2, drop_level=False)
NW_1.loc[:, 'thickness'] = thicknesses[:7]
NW_2.loc[:, 'thickness'] = thicknesses

NW_1_avg = 1000* np.sum(NW_1.loc[:, 'thickness']*NW_1.loc[:, 'D50']) / np.sum(NW_1.loc[:, 'thickness'].values)
NW_2_avg = 1000* np.sum(NW_2.iloc[:7].loc[:,'thickness']*NW_2.iloc[:7].loc[:, 'D50']) / np.sum(NW_2.iloc[:7].loc[:, 'thickness'].values)

NW_2_FS_avg = 1000* np.sum(NW_2.iloc[:4].loc[:,'thickness']*NW_2.iloc[:4].loc[:, 'D50']) / np.sum(NW_2.iloc[:4].loc[:, 'thickness'].values)
NW_2_CS_avg = 1000* np.sum(NW_2.iloc[4:7].loc[:,'thickness']*NW_2.iloc[4:7].loc[:, 'D50']) / np.sum(NW_2.iloc[4:7].loc[:, 'thickness'].values)

NW_1_FS_avg = 1000* np.sum(NW_1.iloc[:3].loc[:,'thickness']*NW_1.iloc[:3].loc[:, 'D50']) / np.sum(NW_1.iloc[:3].loc[:, 'thickness'].values)
NW_1_CS_avg = 1000* np.sum(NW_1.iloc[3:7].loc[:,'thickness']*NW_1.iloc[3:7].loc[:, 'D50']) / np.sum(NW_1.iloc[3:7].loc[:, 'thickness'].values)


Duck_1 = Duck.loc[Duck['Filename'].str.contains(locations[2])]
Duck_1 = Duck_1.set_index('Depth').sort_index() # set index to depth
Duck_2 = Duck.loc[Duck['Filename'].str.contains(locations[3])]
Duck_2 = Duck_2.set_index('Depth').sort_index() # set index to depth
Duck_1.loc[:, 'thickness'] = thicknesses
Duck_2.loc[:, 'thickness'] = thicknesses

Duck_1_avg = 1000* np.sum(Duck_1.loc[:, 'thickness']*Duck_1.loc[:, 'D50(mm)']) / np.sum(Duck_1.loc[:, 'thickness'].values)
Duck_2_avg = 1000* np.sum(Duck_2.loc[:, 'thickness']*Duck_2.loc[:, 'D50(mm)']) / np.sum(Duck_2.loc[:, 'thickness'].values)

Duck_1_grad_avg = 1000* np.sum(Duck_1.iloc[:8].loc[:,'thickness']*Duck_1.iloc[:8].loc[:, 'D50(mm)']) / np.sum(Duck_1.iloc[:8].loc[:, 'thickness'].values)
Duck_2_grad_avg = 1000* np.sum(Duck_2.iloc[:7].loc[:,'thickness']*Duck_2.iloc[:7].loc[:, 'D50(mm)']) / np.sum(Duck_2.iloc[:7].loc[:, 'thickness'].values)

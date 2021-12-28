# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:52:18 2021

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
    return (int(average), round(math.sqrt(variance), 1))

thicknesses = [2, 2, 4, 6, 6, 6, 6, 6, 6, 6]

#%%#### PLOT VERTICAL SAMPLING

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

vmin = 240
vmax = 395

camsizer_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Data/Duck/Grainsizes//"
input_file = 'Grainsize_data_Duck.xlsx'
output_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Figures/VerticalSampling/"

data = pd.read_excel(camsizer_dir + input_file, engine='openpyxl')
# data = data.drop([0, 1]) # remove tests
# data = data.drop(range(61, 71)) # drop repetition, should be plotted separately later!

depths = []
samples = []
for i, d in enumerate(data['Filename'].values):
    if '_2.'in d:
        depth = int(d.split('_')[-2].replace('mm',''))
    else:        
        depth = int(d.split('_')[-1].split('.')[0].replace('mm',''))
        
    sample = int(d.split('_')[1].split('.')[0].replace('sample',''))
    
    samples.append(sample)
    depths.append(depth)
data['Depth'] = depths
data['Sample'] = samples

depths = [2, 4, 8, 14, 20, 26, 32, 38, 44, 50]

samples = set(data['Sample'].values)

#%% Set up figure 1 for overview of transport event sampling

rows = ['Loc. {}'.format(row) for row in range(1, 4)]
# cols = ['Time {}'.format(col) for col in ['BA', 'DA', 'AA']]
cols = ['Before\nAeolian Transport', 'During\nAeolian Transport', 'After\nAeolian Transport']

plot_loc_row = [0, 1, 2, 0, 1, 2, 0, 1, 2]
plot_loc_col = [0, 0, 0, 1, 1, 1, 2, 2, 2]

# Prepare coloring based on median value
norm  = colors.Normalize(vmin=vmin, vmax=vmax)
scalarMap = cmx.ScalarMappable(norm=norm, cmap='hot_r')
data['colorval'] = data['D50(mm)']
data['colorval'] = [scalarMap.to_rgba(d*1000) for d in data['colorval']]

fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 10))
fig.subplots_adjust(right=0.85)
fig.subplots_adjust(left=0.14)
fig.subplots_adjust(bottom=0.1)
fig.subplots_adjust(top=0.9)

samples1 = [s for s in samples if s < 10]

for i, s in enumerate(samples1):
    print(s)
    # Select data belonging to sample
    location = data.loc[data['Filename'].str.contains('sample' + str(s) + '_')]
    location_sort = location.set_index('Depth').sort_index() # set index to depth
    location_plot = location_sort[['D16(mm)', 'D25(mm)', 'D50(mm)', 'D75(mm)', 'D84(mm)']].values.T * 1000 # convert to values in mm's
    colorcodes = location_sort['colorval']

    # include vertical grid
    axs[plot_loc_row[i], plot_loc_col[i]].xaxis.grid(True)        
    
    # Plot boxplot in subplot
    c = 'red'
    bplot = axs[plot_loc_row[i], plot_loc_col[i]].boxplot(location_plot, vert=False, showfliers=False, whis=5.0, widths=0.5, patch_artist = True, boxprops=dict(color='black'), medianprops=dict(linewidth=2.0, color='grey'))
                                                  # boxprops=dict(facecolor=c, color=c),
                                                  # capprops=dict(color=c),
                                                  # whiskerprops=dict(color=c),
                                                  # flierprops=dict(color=c, markeredgecolor=c),
                                                  # )
    for patch, color in zip(bplot['boxes'], colorcodes):
        patch.set_facecolor(color)
        
    # Set limits and ticks of subplot    
    axs[plot_loc_row[i], plot_loc_col[i]].set_xlim(175, 850)
    axs[plot_loc_row[i], plot_loc_col[i]].set_ylim(0, 12.5)
    axs[plot_loc_row[i], plot_loc_col[i]].set_yticks(range(1,11))
    axs[plot_loc_row[i], plot_loc_col[i]].set_yticklabels(depths)
    
    axs[plot_loc_row[i], plot_loc_col[i]].tick_params(axis='both', which='major', labelsize=16)
    axs[plot_loc_row[i], plot_loc_col[i]].tick_params(axis='both', which='minor', labelsize=16)

    # Add average and standard deviation of median grain sizes
    avg, stddev = weighted_avg_and_std(location_sort.loc[:, 'D50(mm)']*1000, thicknesses[:len(location_sort.loc[:, 'D50(mm)'])])
    props = dict(facecolor='white', edgecolor = 'white')
    axs[plot_loc_row[i], plot_loc_col[i]].axhline(y=10.65, color = 'grey', linewidth=0.5)
    axs[plot_loc_row[i], plot_loc_col[i]].text(210, 11.85,'$\u00f8_{50}$ = ' + str(avg) + '\t  $\u03C3_{50}$ = ' + str(stddev), fontsize=14, bbox = props)
                
# Flip axes so depth is shown correctly    
plt.gca().invert_yaxis()    
plt.rcParams.update({'axes.titlesize': 'xx-large'})
# plt.rcParams.update({'axes.labelsize': '16'})

# Set axes labels and only show them for outer axes    
# plt.setp(axs.flat, xlabel='Grain size ($\mu$m)', ylabel='Depth (mm)')    
for ax in axs.flat:
    ax.set_xlabel('Grain size ($\mu$m)', fontsize=16)
    ax.set_ylabel('Depth (mm)', fontsize=16)   
    ax.label_outer()
    
# Add column header
for ax, col in zip(axs[0], cols):
    ax.set_title(col)

# Add row headers
pad = 5
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='xx-large', ha='right', va='center')
    
cax = fig.add_axes([0.88, 0.10, 0.02, 0.8])
fig.colorbar(scalarMap, cax=cax, orientation='vertical')
cax.tick_params(labelsize=18) 
cax.set_ylabel('Median grain size ($\mu$m)', fontsize = 16)    
        
# Finish layout
# fig1.tight_layout()
plt.show()

# Save as file
file_name = 'Grain_size_vertical'
plt.savefig(output_dir + file_name + '_Dunex_boxplot_aeolian.png')
# plt.close()

#%% Set up figure 2 for overview of detailed sampling after tranport event

depths2 = [2, 4, 8]

cols = ['Loc. {}'.format(col) for col in [1, 2, 3, 10, 11, 12, 13, 14, 15, 16]]
# rows = ['Time {}'.format(row) for row in ['C']]

plot_loc_row = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
plot_loc_col = [2, 1, 4, 0, 1, 3, 4, 0, 2, 3]

# Prepare coloring based on median value
norm2  = colors.Normalize(vmin=vmin, vmax=vmax)
scalarMap2 = cmx.ScalarMappable(norm=norm2, cmap='hot_r')
data['colorval2'] = data['D50(mm)']
data['colorval2'] = [scalarMap2.to_rgba(d*1000) for d in data['colorval2']]


fig2, axs = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(10,3.5))
plt.subplots_adjust(wspace = 0.7)
fig2.subplots_adjust(right=0.8)
fig2.subplots_adjust(left=0.1)
fig2.subplots_adjust(bottom=0.1)
fig2.subplots_adjust(top=0.95)

samples2 = [s for s in samples if s > 6 and s < 17]

for i, s in enumerate(samples2):
    print(i, s)
    # Select data belonging to sample
    location = data.loc[data['Filename'].str.contains('sample' + str(s) + '_')]
    location_sort = location.set_index('Depth').sort_index() # set index to depth
    location_plot = location_sort[['D16(mm)', 'D25(mm)', 'D50(mm)', 'D75(mm)', 'D84(mm)']].iloc[0:len(depths2)].values.T * 1000 # convert to values in mm's
    colorcodes2 = location_sort['colorval2']

    for j, layer in enumerate(depths2):
        if j == 0:
            upper_boundary = 0
        else: 
            upper_boundary = -1 * depths2[j-1]
        lower_boundary = -1 * depths2[j]
        
        axs[plot_loc_row[i], plot_loc_col[i]].fill_between([0., 1.], [upper_boundary, upper_boundary],
                             [lower_boundary, lower_boundary],
                             facecolor=colorcodes2.iloc[j], edgecolor = 'k')    
    
    # Hide the right and top spines
    axs[plot_loc_row[i], plot_loc_col[i]].spines['right'].set_visible(False)
    axs[plot_loc_row[i], plot_loc_col[i]].spines['top'].set_visible(False)
    axs[plot_loc_row[i], plot_loc_col[i]].spines['bottom'].set_visible(False)
    axs[plot_loc_row[i], plot_loc_col[i]].spines['left'].set_visible(False)
    axs[plot_loc_row[i], plot_loc_col[i]].tick_params(bottom=False, labelbottom=False, left=True, labelleft=True)
    axs[plot_loc_row[i], plot_loc_col[i]].tick_params(labelsize=14) 
    
    axs[plot_loc_row[i],plot_loc_col[i]].set_yticks([-2,-4,-8])
    axs[plot_loc_row[i],plot_loc_col[i]].set_yticklabels(depths2)
        
    axs[plot_loc_row[i], plot_loc_col[i]].tick_params(axis='both', which='major', labelsize=16)
    axs[plot_loc_row[i], plot_loc_col[i]].tick_params(axis='both', which='minor', labelsize=16)
    
cax = fig2.add_axes([0.88, 0.10, 0.02, 0.8])
fig2.colorbar(scalarMap2, cax=cax, orientation='vertical')
cax.tick_params(labelsize=18) 
cax.set_ylabel('Median grain size ($\mu$m)', fontsize = 16)    

# Flip axes so depth is shown correctly    
# plt.gca().invert_yaxis()  
plt.rcParams.update({'axes.titlesize': 'xx-large'})

# Set axes labels and only show them for outer axes    
# plt.setp(axs.flat, xlabel='Grain size ($\mu$m)', ylabel='Depth (mm)')    
for ax in axs.flat:
    # ax.set_xlabel('Grain size ($\mu$m)', fontsize=14)
    ax.set_ylabel('Depth (mm)', fontsize=16)   
    # ax.label_outer()
    
# Finish layout
# fig2.tight_layout()
plt.show()

# Save as file
file_name = 'Grain_size_vertical'
plt.savefig(output_dir + file_name + '_Dunex_boxplot_detailed.png')
# plt.close()

#%% Set up figure 3 for overview of sampling of effect high water

cols = ['Location {}'.format(col) for col in [10, 11, 1, 12]]
# rows = ['Time {}'.format(row) for row in ['BT', 'AT']]
rows = ['Before\nHW', 'After\nHW']

samples3 = [s for s in samples if s > 16 or s == 7 or s < 13 and s > 9]

depths = [2, 4, 8, 14, 20, 26]

plot_loc_row = [0, 0, 0, 0, 1, 1, 1, 1]
plot_loc_col = [2, 0, 1, 3, 3, 2, 1, 0]

# Prepare coloring based on median value
norm3  = colors.Normalize(vmin=vmin, vmax=vmax)
scalarMap3 = cmx.ScalarMappable(norm=norm3, cmap='hot_r')
data['colorval3'] = data['D50(mm)']
data['colorval3'] = [scalarMap3.to_rgba(d*1000) for d in data['colorval3']]

fig1, axs = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(12,5))
fig1.subplots_adjust(right=0.85)
fig1.subplots_adjust(left=0.17)
fig1.subplots_adjust(bottom=0.2)
fig1.subplots_adjust(top=0.9)

for i, s in enumerate(samples3):
    print(i, s)
    # Select data belonging to sample
    location = data.loc[data['Filename'].str.contains('sample' + str(s) + '_')]
    location_sort = location.set_index('Depth').sort_index() # set index to depth
    location_plot = location_sort[['D16(mm)', 'D25(mm)', 'D50(mm)', 'D75(mm)', 'D84(mm)']].iloc[0:len(depths)].values.T * 1000 # convert to values in mm's
    colorcodes = location_sort['colorval']

    # include vertical grid
    axs[plot_loc_row[i], plot_loc_col[i]].xaxis.grid(True)        
    
    # Plot boxplot in subplot
    c = 'red'
    bplot = axs[plot_loc_row[i], plot_loc_col[i]].boxplot(location_plot, vert=False, showfliers=False, whis=5.0, widths=0.5, patch_artist = True, boxprops=dict(color='black'), medianprops=dict(linewidth=2.0, color='grey'))
                                                  # boxprops=dict(facecolor=c, color=c),
                                                  # capprops=dict(color=c),
                                                  # whiskerprops=dict(color=c),
                                                  # flierprops=dict(color=c, markeredgecolor=c),
                                                  # )
    for patch, color in zip(bplot['boxes'], colorcodes):
        patch.set_facecolor(color)
        
    # Set limits and ticks of subplot    
    axs[plot_loc_row[i], plot_loc_col[i]].set_xlim(150, 700)
    axs[plot_loc_row[i], plot_loc_col[i]].set_ylim(0, len(depths)+0.5)
    axs[plot_loc_row[i], plot_loc_col[i]].set_yticks(range(1,len(depths)+1))
    axs[plot_loc_row[i], plot_loc_col[i]].set_yticklabels(depths)

    axs[plot_loc_row[i], plot_loc_col[i]].tick_params(axis='both', which='major', labelsize=16)
    axs[plot_loc_row[i], plot_loc_col[i]].tick_params(axis='both', which='minor', labelsize=16)

    
# Flip axes so depth is shown correctly    
plt.gca().invert_yaxis()    
plt.rcParams.update({'axes.titlesize': 'xx-large'})    

# Set axes labels and only show them for outer axes    
# plt.setp(axs.flat, xlabel='Grain size ($\mu$m)', ylabel='Depth (mm)')    
for ax in axs.flat:
    ax.set_xlabel('Grain size ($\mu$m)', fontsize=16)
    ax.set_ylabel('Depth (mm)', fontsize=16)   
    ax.label_outer()
    
# Add column header
for ax, col in zip(axs[0], cols):
    ax.set_title(col, fontsize = 16)

# Add row headers
pad = 5
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='xx-large', ha='right', va='center')
        
# Add colorbar    
cax = fig1.add_axes([0.87, 0.2, 0.02, 0.7])
fig1.colorbar(scalarMap3, cax=cax, orientation='vertical')
cax.tick_params(labelsize=18) 
cax.set_ylabel('Median grain size ($\mu$m)', fontsize = 16)       
    
# Finish layout
# fig1.tight_layout()
plt.show()

# Save as file
file_name = 'Grain_size_vertical'
plt.savefig(output_dir + file_name + '_Dunex_boxplot_marine.png')
# plt.close()

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

grainsizes_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Data/Noordwijk/Grainsizes/"
grainsizes_file = 'Grainsize_data_Noordwijk.xlsx'
output_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Figures/VerticalSampling/"

data = pd.read_excel(grainsizes_dir + grainsizes_file, engine='openpyxl')

data['Date'].fillna(method='ffill', inplace=True)
data['Type'].fillna(method='ffill', inplace=True)
data['Location'].fillna(method='ffill', inplace=True)
data = data.set_index(['Type', 'Date', 'Location'])

data['Depth'] = [int(d.replace(' mm', '')) for d in data['Depth']]

#%% Set up figure 1 for february 5, 2020

sample_code = 'V'   
date = 52
date_str = '5 February'
locations = ['_01', '_02', '_03', '_04', '_05', '_06']

depths = [2, 4, 8, 14, 20, 26, 32, 38, 44, 50]

rows = ['Loc. {}'.format(row) for row in range(1, 4)]
# cols = ['Time {}'.format(col) for col in ['BT', 'AT/BA']]
cols = ['Before High water', 'After High Water']

plot_loc_col = [0, 0, 0, 1, 1, 1]
plot_loc_row = [0, 1, 2, 0, 1, 2]

sel_date2 = data.xs(date, level=1, drop_level=False)

# Prepare coloring based on median value
norm  = colors.Normalize(vmin=vmin, vmax=vmax)
scalarMap = cmx.ScalarMappable(norm=norm, cmap='hot_r')
sel_date2.loc[:,'colorval'] = sel_date2['D50']
sel_date2['colorval'] = [scalarMap.to_rgba(d*1000) for d in sel_date2['colorval']]

fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 10))
fig.subplots_adjust(right=0.85)
fig.subplots_adjust(left=0.14)
fig.subplots_adjust(bottom=0.1)
fig.subplots_adjust(top=0.9)

for i, s in enumerate(locations):
    print(s)
    # Select data belonging to sample
    location = sel_date2.xs(s, level=2, drop_level=False)
    print(np.max(location['D50']) - np.min(location['D50']))
    location_sort = location.set_index('Depth').sort_index() # set index to depth
    location_plot = location_sort[['D16', 'D25', 'D50', 'D75', 'D84']].values.T * 1000 # convert to values in mm's
    colorcodes = location_sort['colorval']

    # include vertical grid
    axs[plot_loc_row[i], plot_loc_col[i]].xaxis.grid(True)      
    
    # Add average and standard deviation of median grain sizes
    avg, stddev = weighted_avg_and_std(location_sort.loc[:, 'D50']*1000, thicknesses[:len(location_sort.loc[:, 'D50'])])
    props = dict(facecolor='white', edgecolor = 'white')
    axs[plot_loc_row[i], plot_loc_col[i]].axvline(x=avg, color = 'dimgrey', linewidth=2)
    #axs[plot_loc_row[i], plot_loc_col[i]].axhline(y=0.15, color = 'grey', linewidth=0.5)
    axs[plot_loc_row[i], plot_loc_col[i]].text(avg+20, -0.52,'$\u00f8_{50}$ = ' + str(avg), fontsize=14, bbox = props) #+ '\t  $\u03C3_{50}$ = ' + str(stddev)
        
    # Plot boxplot in subplot
    bplot = axs[plot_loc_row[i], plot_loc_col[i]].boxplot(location_plot, vert=False, showfliers=False, whis=50.0, widths=0.5, patch_artist = True, boxprops=dict(color='black'), medianprops=dict(linewidth=2.0, color='grey'))
                                                  # boxprops=dict(facecolor=c, color=c),
                                                  # capprops=dict(color=c),
                                                  # whiskerprops=dict(color=c),
                                                  # flierprops=dict(color=c, markeredgecolor=c),
                                                  # )

    for patch, color in zip(bplot['boxes'], colorcodes):
        patch.set_facecolor(color) 

    # Set limits and ticks of subplot    
    axs[plot_loc_row[i], plot_loc_col[i]].set_xlim(150, 600)
    axs[plot_loc_row[i], plot_loc_col[i]].set_ylim(-1.7, 10.5)
    axs[plot_loc_row[i], plot_loc_col[i]].set_yticks(range(1,11))
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
    ax.set_title(col)

# Add row headers
pad = 5
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='xx-large', ha='right', va='center')
    
# Delete empty axes
for i, ax in enumerate([(0,2), (1,2) , (2,2)]):
    # if i == 0 or i == 1:
    axs[ax].xaxis.set_visible(False)
    plt.setp(axs[ax].spines['bottom'], visible=False)
    # if i == 1 or i == 3:    
    axs[ax].tick_params(left=False, labelleft=False)
    plt.setp(axs[ax].spines['left'], visible=False) 
    
    plt.setp(axs[ax].spines['right'], visible=False)
    plt.setp(axs[ax].spines['top'], visible=False)
    
cax = fig.add_axes([0.67, 0.10, 0.02, 0.80])
fig.colorbar(scalarMap, cax=cax, orientation='vertical')
cax.tick_params(labelsize=18) 
cax.set_ylabel('Median grain size ($\mu$m)', fontsize = 18)   
        
# Finish layout
# fig.tight_layout()
plt.show()

# Save as file
file_name = 'Grain_size_vertical'
plt.savefig(output_dir + file_name + '_Noordwijk_boxplot_marine.png')
# plt.close()

#%% Set up figure 2 for january 21, 2021

sample_code = 'V'   
date = 2101
date_str = '21 January'
locations = ['_01', '_02', '_03', '_04', '_05', '_06', '_07', '_08', '_09']

depths = [2, 4, 8, 14, 20, 26, 32, 38, 44, 50]

rows = ['Loc. {}'.format(row) for row in range(1, 4)]
# cols = ['Time {}'.format(col) for col in ['BT', 'AT/BA', 'AA']]
cols = ['Before High Water', 'After High Water/\nBefore Aeolian Transport', 'After Aeolian Transport']

plot_loc_col = [0, 0, 0, 1, 1, 1, 2, 2, 2]
plot_loc_row = [0, 1, 2, 2, 1, 0, 0, 1, 2]

sel_date = data.xs(date, level=1, drop_level=False)

# Prepare coloring based on median value
sel_date.loc[:,'colorval'] = sel_date['D50']
sel_date['colorval'] = [scalarMap.to_rgba(d*1000) for d in sel_date['colorval']]

fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 10))
fig.subplots_adjust(right=0.85)
fig.subplots_adjust(left=0.14)
fig.subplots_adjust(bottom=0.1)
fig.subplots_adjust(top=0.9)

for i, s in enumerate(locations):
    print(s)
    # Select data belonging to sample
    location = sel_date.xs(s, level=2, drop_level=False)
    print(np.max(location['D50']) - np.min(location['D50']))
    location_sort = location.set_index('Depth').sort_index() # set index to depth
    location_plot = location_sort[['D16', 'D25', 'D50', 'D75', 'D84']].values.T * 1000 # convert to values in mm's
    colorcodes = location_sort['colorval']

    # include vertical grid
    axs[plot_loc_row[i], plot_loc_col[i]].xaxis.grid(True) 
    
    # Add average and standard deviation of median grain sizes
    avg, stddev = weighted_avg_and_std(location_sort.loc[:, 'D50']*1000, thicknesses[:len(location_sort.loc[:, 'D50'])])
    props = dict(facecolor='white', edgecolor = 'white')
    axs[plot_loc_row[i], plot_loc_col[i]].axvline(x=avg, color = 'dimgrey', linewidth=2)
    #axs[plot_loc_row[i], plot_loc_col[i]].axhline(y=0.15, color = 'grey', linewidth=0.5)
    axs[plot_loc_row[i], plot_loc_col[i]].text(avg+20, -0.52,'$\u00f8_{50}$ = ' + str(avg), fontsize=14, bbox = props) #+ '\t  $\u03C3_{50}$ = ' + str(stddev)
            
    # Plot boxplot in subplot
    bplot = axs[plot_loc_row[i], plot_loc_col[i]].boxplot(location_plot, vert=False, showfliers=False, whis=5.0, widths=0.5, patch_artist = True, boxprops=dict(color='black'), medianprops=dict(linewidth=2.0, color='grey'))
                                                  # boxprops=dict(facecolor=c, color=c),
                                                  # capprops=dict(color=c),
                                                  # whiskerprops=dict(color=c),
                                                  # flierprops=dict(color=c, markeredgecolor=c),
                                                  # )

    for patch, color in zip(bplot['boxes'], colorcodes):
        patch.set_facecolor(color)
        
    # Set limits and ticks of subplot    
    axs[plot_loc_row[i], plot_loc_col[i]].set_xlim(150, 450)
    axs[plot_loc_row[i], plot_loc_col[i]].set_ylim(-1.7, 10.5)
    axs[plot_loc_row[i], plot_loc_col[i]].set_yticks(range(1,11))
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
    ax.set_title(col)

# Add row headers
pad = 5
for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='xx-large', ha='right', va='center')
        
cax = fig.add_axes([0.87, 0.10, 0.02, 0.80])
fig.colorbar(scalarMap, cax=cax, orientation='vertical')
cax.tick_params(labelsize=18) 
cax.set_ylabel('Median grain size ($\mu$m)', fontsize = 18)       
    
# Finish layout
# fig.tight_layout()
plt.show()

# Save as file
file_name = 'Grain_size_vertical'
plt.savefig(output_dir + file_name + '_Noordwijk_boxplot_aeolian.png')
# plt.close()



#%% Set up figure 3 for december 20, 2020

# sample_code = 'V'   
# date = 2012
# date_str = '20 December'
# locations = ['_01', '_02', '_03', '_04', '_05', '_06']

# depths = [2, 4, 8, 14, 20, 26, 32, 38, 44, 50]

# rows = ['Loc1 {}'.format(row) for row in range(1, 4)]
# cols = ['Time {}'.format(col) for col in ['A', 'B', 'C']]

# plot_loc_col = [0, 0, 0, 1, 1, 1]
# plot_loc_row = [0, 1, 2, 0, 1, 2]

# fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(12, 8))
# sel_date = data.xs(date, level=1, drop_level=False)

# for i, s in enumerate(locations):
#     print(s)
#     # Select data belonging to sample
#     location = sel_date.xs(s, level=2, drop_level=False)
#     location_sort = location.set_index('Depth').sort_index() # set index to depth
#     location_plot = location_sort[['D10', 'D25', 'D50', 'D75', 'D90']].values.T * 1000 # convert to values in mm's
#     colorcodes = location_sort['colorval']

#     # include vertical grid
#     axs[plot_loc_row[i], plot_loc_col[i]].xaxis.grid(True) 
    
#     # Plot boxplot in subplot
#     bplot = axs[plot_loc_row[i], plot_loc_col[i]].boxplot(location_plot, vert=False, showfliers=False, whis=5.0, widths=0.5, patch_artist = True, boxprops=dict(color='black'), medianprops=dict(linewidth=2.0, color='grey'))
#                                                   # boxprops=dict(facecolor=c, color=c),
#                                                   # capprops=dict(color=c),
#                                                   # whiskerprops=dict(color=c),
#                                                   # flierprops=dict(color=c, markeredgecolor=c),
#                                                   # )

#     for patch, color in zip(bplot['boxes'], colorcodes):
#         patch.set_facecolor(color)
        
#     # Set limits and ticks of subplot    
#     axs[plot_loc_row[i], plot_loc_col[i]].set_xlim(150, 600)
#     axs[plot_loc_row[i], plot_loc_col[i]].set_ylim(0, 10.5)
#     axs[plot_loc_row[i], plot_loc_col[i]].set_yticks(range(1,11))
#     axs[plot_loc_row[i], plot_loc_col[i]].set_yticklabels(depths)
    
# # Flip axes so depth is shown correctly    
# plt.gca().invert_yaxis()    

# # Set axes labels and only show them for outer axes    
# plt.setp(axs.flat, xlabel='Grain size ($\mu$m)', ylabel='Depth (mm)')    
# for ax in axs.flat:
#     ax.label_outer()
    
# # Add column header
# for ax, col in zip(axs[0], cols):
#     ax.set_title(col)

# # Add row headers
# pad = 5
# for ax, row in zip(axs[:,0], rows):
#     ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
#                 xycoords=ax.yaxis.label, textcoords='offset points',
#                 size='large', ha='right', va='center')
        
# # Finish layout
# fig.tight_layout()
# plt.show()

# # Save as file
# file_name = 'Grain_size_vertical'
# plt.savefig(output_dir + file_name + '_Noordwijk_boxplot_20122020.png')
# # plt.close()


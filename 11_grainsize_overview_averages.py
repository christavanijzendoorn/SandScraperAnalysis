# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:09:07 2021

@author: cijzendoornvan
"""
#%%#### PLOT VERTICAL SAMPLING
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (int(average), round(math.sqrt(variance), 1))

thicknesses = [2, 2, 4, 6, 6, 6, 6, 6, 6, 6]

#%% Prep dataframe and set mulit index for selection of data

arrays = [
    ["Waldport", "Waldport", "Waldport", "Noordwijk", "Noordwijk", "Noordwijk", "Noordwijk", "Noordwijk", "Noordwijk", "Duck", "Duck", "Duck", "Duck"],
    ["23-08-2021", "23-08-2021", "23-08-2021", "05-02-2020", "05-02-2020", "05-02-2020", "21-01-2021", "21-01-2021", "21-01-2021", "01-09-2021", "01-09-2021", "01-09-2021", "02-09-2021"],
    ["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3", "1"],
]

tuples = list(zip(*arrays))
idx = pd.MultiIndex.from_tuples(tuples, names=["Field site", "Date", 'Sample location'])

arrays_cols = [
    ["Average D50", "Average D50", "Average D50", "Average D50", "Std dev D50", "Std dev D50", "Std dev D50", "Std dev D50"],
    ["BHW", "AHW/BAT", "DAT", "AAT", "BHW", "AHW/BAT", "DAT", "AAT"],
]

tuples_cols = list(zip(*arrays_cols))
cols_idx = pd.MultiIndex.from_tuples(tuples_cols)

df = pd.DataFrame(np.zeros((13, 8)), index=idx, columns=cols_idx)

#%% Calculate Waldport data
camsizer_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/OREGON/Fieldwork/Camsizer/"
input_file = 'camsizer_outputs.xlsx'
outputdir = 'C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Figures/'

data = pd.read_excel(camsizer_dir + input_file, engine='openpyxl')
samples = ['A2', 'A3', 'A4', 'B2', 'B3', 'B4', 'C4']

# Select subsection relating to one sampling occassion
WP_A2 = data.loc[data['Filename'].str.contains(samples[0])]
WP_A3 = data.loc[data['Filename'].str.contains(samples[1])]
WP_A4 = data.loc[data['Filename'].str.contains(samples[2])]
WP_B2 = data.loc[data['Filename'].str.contains(samples[3])]
WP_B3 = data.loc[data['Filename'].str.contains(samples[4])]
WP_B4 = data.loc[data['Filename'].str.contains(samples[5])]
WP_C4 = data.loc[data['Filename'].str.contains(samples[6])]

# Calculate weighted average and standard deviation for each subsection
df['Average D50', 'BHW'].loc['Waldport','23-08-2021','1'], df['Std dev D50','BHW']['Waldport','23-08-2021','1'] = weighted_avg_and_std(WP_A2.loc[:, 'D50(mm)']*1000, thicknesses[:8])
df['Average D50', 'AHW/BAT'].loc['Waldport','23-08-2021','1'], df['Std dev D50','AHW/BAT']['Waldport','23-08-2021','1'] = weighted_avg_and_std(WP_B2.iloc[:8].loc[:, 'D50(mm)']*1000, thicknesses[:8])
df['Average D50', 'BHW'].loc['Waldport','23-08-2021','2'], df['Std dev D50','BHW']['Waldport','23-08-2021','2'] = weighted_avg_and_std(WP_A3.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'AHW/BAT'].loc['Waldport','23-08-2021','2'], df['Std dev D50','AHW/BAT']['Waldport','23-08-2021','2'] = weighted_avg_and_std(WP_B3.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'BHW'].loc['Waldport','23-08-2021','3'], df['Std dev D50','BHW']['Waldport','23-08-2021','3'] = weighted_avg_and_std(WP_A4.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'AHW/BAT'].loc['Waldport','23-08-2021','3'], df['Std dev D50','AHW/BAT']['Waldport','23-08-2021','3'] = weighted_avg_and_std(WP_B4.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'AAT'].loc['Waldport','23-08-2021','3'], df['Std dev D50','AAT']['Waldport','23-08-2021','3'] = weighted_avg_and_std(WP_C4.loc[:, 'D50(mm)']*1000, thicknesses)

#%% Calculate Noordwijk data 21 January 2021
grainsizes_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/SCANEX/Data/Sand_sampling/"
grainsizes_file = 'SampleData_grainsizes_211004.xlsx'
output_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/SCANEX/Data_analysis/Figures"

data = pd.read_excel(grainsizes_dir + grainsizes_file, engine='openpyxl')
data['Date'].fillna(method='ffill', inplace=True)
data['Type'].fillna(method='ffill', inplace=True)
data['Location'].fillna(method='ffill', inplace=True)
data = data.set_index(['Type', 'Date', 'Location'])
data['Depth'] = [int(d.replace(' mm', '')) for d in data['Depth']]

date = 2101
locations = ['_01', '_02', '_03', '_04', '_05', '_06', '_07', '_08', '_09']
sel_date = data.xs(date, level=1, drop_level=False)

# Select subsection relating to one sampling occassion
NW_1_BHW = sel_date.xs(locations[0], level=2, drop_level=False)
NW_2_BHW = sel_date.xs(locations[1], level=2, drop_level=False)
NW_3_BHW = sel_date.xs(locations[2], level=2, drop_level=False)
NW_3_AHW = sel_date.xs(locations[3], level=2, drop_level=False)
NW_2_AHW = sel_date.xs(locations[4], level=2, drop_level=False)
NW_1_AHW = sel_date.xs(locations[5], level=2, drop_level=False)
NW_1_AAT = sel_date.xs(locations[6], level=2, drop_level=False)
NW_2_AAT = sel_date.xs(locations[7], level=2, drop_level=False)
NW_3_AAT = sel_date.xs(locations[8], level=2, drop_level=False)

# Calculate weighted average and standard deviation for each subsection
df['Average D50', 'BHW'].loc['Noordwijk','21-01-2021','1'], df['Std dev D50','BHW']['Noordwijk','21-01-2021','1'] = weighted_avg_and_std(NW_1_BHW.loc[:, 'D50']*1000, thicknesses)
df['Average D50', 'AHW/BAT'].loc['Noordwijk','21-01-2021','1'], df['Std dev D50','AHW/BAT']['Noordwijk','21-01-2021','1'] = weighted_avg_and_std(NW_1_AHW.loc[:, 'D50']*1000, thicknesses)
df['Average D50', 'AAT'].loc['Noordwijk','21-01-2021','1'], df['Std dev D50','AAT']['Noordwijk','21-01-2021','1'] = weighted_avg_and_std(NW_1_AAT.loc[:, 'D50']*1000, thicknesses)
df['Average D50', 'BHW'].loc['Noordwijk','21-01-2021','2'], df['Std dev D50','BHW']['Noordwijk','21-01-2021','2'] = weighted_avg_and_std(NW_2_BHW.loc[:, 'D50']*1000, thicknesses)
df['Average D50', 'AHW/BAT'].loc['Noordwijk','21-01-2021','2'], df['Std dev D50','AHW/BAT']['Noordwijk','21-01-2021','2'] = weighted_avg_and_std(NW_2_AHW.loc[:, 'D50']*1000, thicknesses)
df['Average D50', 'AAT'].loc['Noordwijk','21-01-2021','2'], df['Std dev D50','AAT']['Noordwijk','21-01-2021','2'] = weighted_avg_and_std(NW_2_AAT.loc[:, 'D50']*1000, thicknesses)
df['Average D50', 'BHW'].loc['Noordwijk','21-01-2021','3'], df['Std dev D50','BHW']['Noordwijk','21-01-2021','3'] = weighted_avg_and_std(NW_3_BHW.loc[:, 'D50']*1000, thicknesses[:9])
df['Average D50', 'AHW/BAT'].loc['Noordwijk','21-01-2021','3'], df['Std dev D50','AHW/BAT']['Noordwijk','21-01-2021','3'] = weighted_avg_and_std(NW_3_AHW.iloc[:9].loc[:, 'D50']*1000, thicknesses[:9])
df['Average D50', 'AAT'].loc['Noordwijk','21-01-2021','3'], df['Std dev D50','AAT']['Noordwijk','21-01-2021','3'] = weighted_avg_and_std(NW_3_AAT.iloc[:9].loc[:, 'D50']*1000, thicknesses[:9])

#%% Calculate Noordwijk data 5 February 2020
date = 52
locations = ['_01', '_02', '_03', '_04', '_05', '_06']
sel_date2 = data.xs(date, level=1, drop_level=False)

# Select subsection relating to one sampling occassion
NW2_1_BHW = sel_date2.xs(locations[0], level=2, drop_level=False)
NW2_2_BHW = sel_date2.xs(locations[1], level=2, drop_level=False)
NW2_3_BHW = sel_date2.xs(locations[2], level=2, drop_level=False)
NW2_3_AHW = sel_date2.xs(locations[3], level=2, drop_level=False)
NW2_2_AHW = sel_date2.xs(locations[4], level=2, drop_level=False)
NW2_1_AHW = sel_date2.xs(locations[5], level=2, drop_level=False)

# Calculate weighted average and standard deviation for each subsection
df['Average D50', 'BHW'].loc['Noordwijk','05-02-2020','1'], df['Std dev D50','BHW']['Noordwijk','05-02-2020','1'] = weighted_avg_and_std(NW2_1_BHW.loc[:, 'D50']*1000, thicknesses)
df['Average D50', 'BHW'].loc['Noordwijk','05-02-2020','2'], df['Std dev D50','BHW']['Noordwijk','05-02-2020','2'] = weighted_avg_and_std(NW2_2_BHW.loc[:, 'D50']*1000, thicknesses)
df['Average D50', 'BHW'].loc['Noordwijk','05-02-2020','3'], df['Std dev D50','BHW']['Noordwijk','05-02-2020','3'] = weighted_avg_and_std(NW2_3_BHW.loc[:, 'D50']*1000, thicknesses)
df['Average D50', 'AHW/BAT'].loc['Noordwijk','05-02-2020','3'], df['Std dev D50','AHW/BAT']['Noordwijk','05-02-2020','3'] = weighted_avg_and_std(NW2_3_AHW.loc[:, 'D50']*1000, thicknesses)
df['Average D50', 'AHW/BAT'].loc['Noordwijk','05-02-2020','2'], df['Std dev D50','AHW/BAT']['Noordwijk','05-02-2020','2'] = weighted_avg_and_std(NW2_2_AHW.loc[:, 'D50']*1000, thicknesses)
df['Average D50', 'AHW/BAT'].loc['Noordwijk','05-02-2020','1'], df['Std dev D50','AHW/BAT']['Noordwijk','05-02-2020','1'] = weighted_avg_and_std(NW2_1_AHW.loc[:, 'D50']*1000, thicknesses)

#%% Duck data
camsizer_dir = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/DUNEX/Camsizer/"
input_file = 'camsizer_outputs.xlsx'
output_dir = 'C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/GRAINSIZE/Figures/'

data = pd.read_excel(camsizer_dir + input_file, engine='openpyxl')

locations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 18]

# Select subsection relating to one sampling occassion
D_1_BAT = data.loc[data['Filename'].str.contains('sample' + str(locations[0]) + '_')]
D_2_BAT = data.loc[data['Filename'].str.contains('sample' + str(locations[1]) + '_')]
D_3_BAT = data.loc[data['Filename'].str.contains('sample' + str(locations[2]) + '_')]
D_1_DAT = data.loc[data['Filename'].str.contains('sample' + str(locations[3]) + '_')]
D_2_DAT = data.loc[data['Filename'].str.contains('sample' + str(locations[4]) + '_')]
D_3_DAT = data.loc[data['Filename'].str.contains('sample' + str(locations[5]) + '_')]
D_1_AAT = data.loc[data['Filename'].str.contains('sample' + str(locations[6]) + '_')]
D_2_AAT = data.loc[data['Filename'].str.contains('sample' + str(locations[7]) + '_')]
D_3_AAT = data.loc[data['Filename'].str.contains('sample' + str(locations[8]) + '_')]
D_1_AHW = data.loc[data['Filename'].str.contains('sample' + str(locations[9]) + '_')]

# Calculate weighted average and standard deviation for each subsection
df['Average D50', 'AHW/BAT'].loc['Duck','01-09-2021','1'], df['Std dev D50','AHW/BAT']['Duck','01-09-2021','1'] = weighted_avg_and_std(D_1_BAT.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'AHW/BAT'].loc['Duck','01-09-2021','2'], df['Std dev D50','AHW/BAT']['Duck','01-09-2021','2'] = weighted_avg_and_std(D_2_BAT.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'AHW/BAT'].loc['Duck','01-09-2021','3'], df['Std dev D50','AHW/BAT']['Duck','01-09-2021','3'] = weighted_avg_and_std(D_3_BAT.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'DAT'].loc['Duck','01-09-2021','1'], df['Std dev D50','DAT']['Duck','01-09-2021','1'] = weighted_avg_and_std(D_1_DAT.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'DAT'].loc['Duck','01-09-2021','2'], df['Std dev D50','DAT']['Duck','01-09-2021','2'] = weighted_avg_and_std(D_2_DAT.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'DAT'].loc['Duck','01-09-2021','3'], df['Std dev D50','DAT']['Duck','01-09-2021','3'] = weighted_avg_and_std(D_3_DAT.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'AAT'].loc['Duck','01-09-2021','1'], df['Std dev D50','AAT']['Duck','01-09-2021','1'] = weighted_avg_and_std(D_1_AAT.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'AAT'].loc['Duck','01-09-2021','2'], df['Std dev D50','AAT']['Duck','01-09-2021','2'] = weighted_avg_and_std(D_2_AAT.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'AAT'].loc['Duck','01-09-2021','3'], df['Std dev D50','AAT']['Duck','01-09-2021','3'] = weighted_avg_and_std(D_3_AAT.loc[:, 'D50(mm)']*1000, thicknesses)
df['Average D50', 'AHW/BAT'].loc['Duck','02-09-2021','1'], df['Std dev D50','AHW/BAT']['Duck','02-09-2021','1'] = weighted_avg_and_std(D_1_AHW.loc[:, 'D50(mm)']*1000, thicknesses[:6])
df['Average D50', 'BHW'].loc['Duck','02-09-2021','1'], df['Std dev D50','BHW']['Duck','02-09-2021','1'] = weighted_avg_and_std(D_1_AAT.loc[:, 'D50(mm)'][:6]*1000, thicknesses[:6])

# Replace zero values for nan
df = df.replace(0, np.nan)

# Prep calculation of overall std devs
def get_averaged_stddev(stddevs):
    stddevs_sq = stddevs**2 
    n = ~np.isnan(stddevs)
    av_stddevs = np.sqrt(stddevs_sq.sum(axis=1)/n.sum(axis=1))

    return av_stddevs

#%% Create Figure with overview of cross-shore gradients
    
times = [1, 2, 2.5, 3]
locs = [1, 2, 3]

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(12, 4))
fig.subplots_adjust(right=0.81)
fig.subplots_adjust(left=0.14)
fig.subplots_adjust(bottom=0.17)
fig.subplots_adjust(top=0.9)
fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.4)

cs = 5

# Calculate and plot data Duck
avg_stddevs_D = get_averaged_stddev(df['Std dev D50'].loc['Duck','01-09-2021', :])
axs[0,0].errorbar(locs, df['Average D50'].loc['Duck','01-09-2021', :].mean(axis=1), marker = 'o', color = 'goldenrod', yerr = avg_stddevs_D, capsize=5, linewidth=2)
axs[0,0].set_title('Duck', fontsize=16)
axs[0,0].set_ylim(245, 365)

# Calculate and plot data Noordwijk 5 February 2020
avg_stddevs_NW1 = get_averaged_stddev(df['Std dev D50'].loc['Noordwijk','05-02-2020', :])
axs[0,1].errorbar(locs, df['Average D50'].loc['Noordwijk','05-02-2020', :].mean(axis=1), marker = 'o', color = 'royalblue', yerr = avg_stddevs_NW1, capsize=5, linewidth=2)
axs[0,1].set_title('Noordwijk 5 February 2020', fontsize=16)
axs[0,1].set_ylim(245, 365)

# Calculate and plot data Noordwijk 21 January 2021
avg_stddevs_NW2 = get_averaged_stddev(df['Std dev D50'].loc['Noordwijk','21-01-2021', :])
axs[1,1].errorbar(locs, df['Average D50'].loc['Noordwijk','21-01-2021', :].mean(axis=1), marker = 'o', color = 'grey', yerr = avg_stddevs_NW2, capsize=5, linewidth=2)
axs[1,1].set_title('Noordwijk 21 January 2021', fontsize=16)
axs[1,1].set_ylim(245, 365)

# Calculate and plot data Waldport
avg_stddevs_WP = get_averaged_stddev(df['Std dev D50'].loc['Waldport','23-08-2021', :])
axs[1,0].errorbar(locs, df['Average D50'].loc['Waldport','23-08-2021', :].mean(axis=1), marker = 'o', color = 'forestgreen', yerr = avg_stddevs_WP, capsize=5, linewidth=2)
axs[1,0].set_title('Waldport', fontsize=16)
axs[1,0].set_ylim(220, 280) 

# Set axes layout
for ax in axs.flat:
    ax.set_xticks(locs)
    ax.set_xticklabels(['1','2','3'], fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

# Include grain size and location (from seaward to landward) labels
axs[1,0].annotate('Grain size ($\mu$m)', xy=(-200, 1.5), xytext=(-3, 55), 
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='xx-large', ha='right', va='center', rotation = 90)   
axs[1,0].annotate('Grain size ($\mu$m)', xy=(-200, 1.5), xytext=(-338, 55), 
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='xx-large', ha='right', va='center', rotation = 90)   

axs[1,0].annotate('Location                                                Location', xy=(-200, 1.5), xytext=(195, -77), 
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='xx-large', ha='right', va='center')   
axs[1,0].annotate('$\u276E$ sea    \t\t\t   land $\u276F$    \t\t\t     $\u276E$ sea    \t\t\t land $\u276F$', xy=(-200, 1.5), xytext=(273, -77), 
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='x-large', ha='right', va='center')  
      
# Save file
file_name = 'Averaged_per_site'
plt.savefig(outputdir + '/VerticalSampling/' + file_name + '.png')
# plt.savefig(outputdir + '/VerticalSampling/' + file_name + '.eps')

        
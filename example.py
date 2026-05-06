# -*- coding: utf-8 -*-
"""
Created on Wed May 6 15:30:58 2026

example script to load in neurons, get projection frequencies, read out targeted regions, and visualize with brainrender

@author: samkr
"""
# %%

#Step 1

#first import necessary packages
import pandas as pd
import numpy as np
#to load and preprocess data
from main import load_data as ld
from main import preprocess_funcs as pf
# %%

#Step 2

#set directory that contains json files of reconstructions (replace with your filepath to jsons)
pathtocelljsons = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\MRN\json"
#set directory for swc files, needed to visualize with brainrender (replace with your filepath to swcs)
pathtocellswc = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\MRN\swc'

# %%

#Step 3

'''
now, we need to load the neurons into a dictionary where the keys are the neuron ID, and values are the full json
'''

jsondict = ld.load_neurons(pathtocelljsons)

# %%

#Step 4

'''
then, this function will look at each node, index into the ccf volume, and annotate each node with its region info at all levels of ccf ontology

this function can take a while to run with a large number of cells, it is recommended to do these first two steps in a separate script
then pickle the resulting dictionary and save to be reopened in a different script when you want to look at frequency information 

before running this line, go into load_data.py and change allen_ccf_10um to be the path to the ccf volume on your machine
'''

ld.get_node_parcellations(jsondict)


# %%

#Step 5

'''
now, use the following code to get a dataframe containing the number of endpoints in each targeted region for each cell
'''

frequencydf = pd.DataFrame()
for cell in jsondict:
    freqseries = ld.get_frequencies_from_dict({cell: jsondict[cell]}, ontlevel='structure') #you can change ontlevel to your desired ontology level
    frequencydf = pd.concat([frequencydf, freqseries], join='outer', axis=1)
#replace NaNs (regions that aren't targeted by the given cell) with 0s, and transpose so rows are cells and columns are regions
frequencynonan = frequencydf.replace(np.nan, 0).T

# %%

#Step 6

'''
then, you can use this function to get a series containing the targeted regions for a given neuron
'''

n097 = pf.get_targeted_regions(frequencynonan, 'N097-709222-DS')
print(n097)

# %%
'''
now, an example showing how to pickle and reload the parcellated neuron dictionary
'''
import pickle
#replace with your desired save location
savefile = r'D:\data\parcellateddict.pkl'
#%%

#write neuron dictionary to pickled object
pickle.dump(jsondict, open(savefile, 'wb'))
#%%
#load as such
pickleddict = pickle.load(open(savefile, 'rb'))

# %%
'''
now you can use pickleddict in Step 5 instead of jsondict
'''

frequencydf2 = pd.DataFrame()
for cell in pickleddict:
    freqseries2 = ld.get_frequencies_from_dict({cell: pickleddict[cell]}, ontlevel='structure') #you can change ontlevel to your desired ontology level
    frequencydf2 = pd.concat([frequencydf2, freqseries2], join='outer', axis=1)
#replace NaNs (regions that aren't targeted by the given cell) with 0s, and transpose so rows are cells and columns are regions
frequencynonan2 = frequencydf2.replace(np.nan, 0).T

print(pf.get_targeted_regions(frequencynonan2, 'N097-709222-DS'))

# %%
'''
okay let's move on to visualization, this will be done using the python package brainglobe and its subpackage brainrender
i'll just show necessary functions to visualize cells basically, extra info can be found at brainglobe's API docs, found here:
https://brainglobe.info/index.html
'''
#import necessary class from brainrender
from brainrender import Scene

#set path to desired neuron to visualize
cellpath = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\MRN\swc\N097-709222-DS.swc"
#%%

#set up brainrender scene with desired atlas
ccf_scene = Scene(atlas_name='allen_mouse_10um', root=True) #root param here will toggle the whole brain mesh on/off

#%%

#by default the root mesh is visualized with a silhouette, that often gets in the way so this code will turn that off
actors = ccf_scene.get_actors()
actors[0]._needs_silhouette = False

#%%
#can use the following function to add brain region meshes to the scene, use one call for each region to set different colors/alphas
ccf_scene.add_brain_region('MRN', silhouette=False, color='purple', alpha=0.2)

#%%

#brainrender skips some nodes when rendering, unsure why, but the following code will create objects to be passed into brainrender.Scene's add function
cell_actors = pf.swap_for_brainrender(cellpath)

for actor in cell_actors:
    ccf_scene.add(actor)
    
ccf_scene.render()

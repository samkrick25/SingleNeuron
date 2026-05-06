import numpy as np
import pandas as pd
import pickle
import os
from sklearn.feature_selection import VarianceThreshold
from warnings import simplefilter
import json
from vedo import Lines

MIDLINEZ = 5750
MIDLINEZ_10UM = 570

def preprocess(df, log1p=True, pct=True):
    '''
    preprocess data however I want to, accepts a pandas.DataFrame with no NaNs
    this is written to have cells as cols and regions as rows i think

    :param df: pandas.DataFrame

    :param log1p: preprocessing option, default True, if False skips log1p scaling

    :param pct: preprocessing option, default True, if False skips cell size scaling

    returns: pandas.DataFrame
    '''
    data_tonorm = df.copy()

    #natural log scale my data, formula of ln(number of terminals + 1) as per Ding et al. 2025
    #and control for cell size
    #default behavior
    if log1p and pct:
        df_log = np.log(data_tonorm+1)
        col_sums = df_log.sum(axis=0)
        df_pct = (df_log.div(col_sums, axis=1))*100
        return df_pct
    
    if log1p and not pct:
        df_log = np.log(data_tonorm+1)
        return df_log
    
    if not log1p  and pct:
        row_sums = df.sum(axis=1)
        df_pct = (df.div(row_sums, axis=0))*100
        return df_pct
    
    if not log1p and not pct:
        print('what are you using this for')
        return
    
def merge_regions(df):
    '''
    merge ipsilateral/contralateral regions into one column
    (GPT assist)

    :param df: DataFrame with frequency information for cells, columns must be 'Ipsilateral [region]' or 'Contralateral [region]'

    returns: DataFrame with shape (n cells, n regions)
    '''
    #this suppresses the PerformanceWarning that pd throws, this function is still running fast
    simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    ipsi_cols = pd.Series(col for col in df.columns if col.startswith('Ipsilateral'))
    contra_cols = pd.Series(col for col in df.columns if col.startswith('Contralateral'))
    ipsi_regions = ipsi_cols.str.replace('Ipsilateral ', '')
    contra_regions = contra_cols.str.replace('Contrlateral ', '')
    all_regions = set(ipsi_regions) | set(contra_regions)
    
    merged_frequency = pd.DataFrame(index=df.index)
    for region in all_regions:
        ipsi_col = f'Ipsilateral {region}'
        contra_col = f'Contralateral {region}'
        ipsi_series = df[ipsi_col] if ipsi_col in df.columns else None
        contra_series = df[contra_col] if contra_col in df.columns else None
        
        #regions that recieve both ipsi and contra projections
        if ipsi_series is not None and contra_series is not None:
            merged_frequency[region] = df[ipsi_col] + df[contra_col]
            
        #regions that recieve only ipsilateral
        if ipsi_series is not None and contra_series is None:
            merged_frequency[region] = df[ipsi_col]
            
        #regions that only recieve contra
        if ipsi_series is None and contra_series is not None:
            merged_frequency[region] = df[contra_col]
    
    return merged_frequency

def get_df_for_region(df, region):
    '''
    get a df containing lateralized frequency data for a given target region

    :params df: full dataframe with columns as regions, lateralized, and rows as neurons

    :params region: target region as str
    '''
    ipsireg = 'Ipsilateral '+region
    contrareg = 'Contralateral '+region
    fullreg = [ipsireg, contrareg]

    regdf = df[fullreg]
    for cell, val in regdf.iterrows():
        iv, cv = val
        if iv == 0 and cv == 0:
            regdf = regdf.drop(cell)

    return regdf

def get_nodes_in_region(cells, *regions, ontlevel='structure', parcellated, kind=None, infunc=False):
    '''
    get list of nodes in a given region(s)

    Parameters
    ----------
    cells : TYPE
        DESCRIPTION.
    *regions : TYPE
        DESCRIPTION.
    ontlevel : str, optional
        Desired Allen CCF ontology level. The default is 'structure'.
    parcellated : TYPE
        DESCRIPTION.
    kind : TYPE, optional
        DESCRIPTION. The default is None.
    infunc : bool, optional
        True if this is called from a different function also using kwargs for the regions parameter. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if infunc:
        regions=regions[0]
        
    match kind:
        case 'bulk':
            nodes = []
            for _, axon in cells.items():
                for node in axon:
                    if parcellated:
                        try: 
                            if node[ontlevel] in regions:
                                nodes.append(node['sampleNumber'])
                        except KeyError:
                            print(node)
                    #if you want not parcellated, then regions has to be the ont id (numbers), if using parcellated can find the region abv
                    if not parcellated:
                        if node['allenId'] in regions:
                            nodes.append(node)
            return nodes
        case 'by_cell':
            cellstonodes = {}
            for cell, axon in cells.items():
                nodes = []
                for node in axon:
                    if parcellated:
                        try:
                            if node[ontlevel] in regions:
                                nodes.append(node['sampleNumber'])
                        except KeyError:
                            print(node)
                    if not parcellated:
                        if node['allenId'] in regions:
                            nodes.append(node)
                cellstonodes[cell] = nodes
            return cellstonodes
                

def get_coords(nodes, dim='all', mirror=False):
    """
    Coordinate getter for a list of nodes contianing x, y, z coordinates
    
    :param nodes: list of dictionaries, each entry should be a dictionary containing 'x', 'y', 'z' coordinates for said node
    :param dim: str, selecting which coordinates to get, default behavior is to return all coords
    :param mirror: bool, default False to not mirror nodes, but will mirror nodes over saggital axis (z in ccfv3)

    Returns: list of coordinates if only one dimension is selected, np.array of lists where each list contains x, y, z coords if all dims are selected
    """
    match dim:
        case 'x':
            x = [node['x'] for node in nodes]
            return x
        case 'y':
            y = [node['y'] for node in nodes]
            return y
        case 'z':
            if mirror:
                for node in nodes:
                    if node['z'] < 5700:
                        diff = 5700-node['z']
                        node['z'] = diff
                z = [node['z'] for node in nodes]
            else:
                z = [node['z'] for node in nodes]            
            return z
        case 'all':
            for node in nodes:
                if mirror:
                    if node['z'] < 5700:
                        diff = 5700 - node['z']
                        node['z'] = 5700+diff
            coords = np.array([[node['x'], node['y'], node['z']] for node in nodes]) 
            return coords
        
# =============================================================================
# def node_coords_getter(cell, dim, mirror=False, *regions):
#     targets = get_target_nodes_list(cell, regions)
#     coords = np.array(get_coords(cell, dim, mirror) for cell in targets if cell)
#     return coords
# =============================================================================

def get_cells_to_region(freqspkl, regionabv, thresh=3):
    '''
    get cell IDs that project to a given region

    Parameters
    ----------
    freqspkl : str
        Path to your pickled frequency DataFrame
    regionabv : TYPE
        Allen CCF v3 abbreviation for your desired region
    thresh : int, optional
        Set threshold for how many endpoints in a region is considered as projecting to that region. The default is 3.

    Returns
    -------
    poscells : list
        Cell IDs that have # endpoints > thresh for your given region.
    negcells : list
        Cell IDs that have # endpoints <= thresh for your given region.

    '''
    freqs = pickle.load(open(freqspkl, 'rb'))

    latmerged = merge_regions(freqs)
    poscells = [cell for cell in latmerged.index.tolist() if latmerged.loc[cell][regionabv] > thresh]
    negcells = [cell for cell in latmerged.index.tolist() if latmerged.loc[cell][regionabv] <= thresh]

    return poscells, negcells

def get_targeted_regions(data, cell):
    '''
    returns a series containing the targeted regions of a given cell, requires data to be a pandas DataFrame where columns are regions and cells are rows
    cell is the name of the cell you want to see target regions for as str 
    '''
    return data.loc[cell, data.columns[data.loc[cell]!=0].tolist()].sort_values(ascending=False)

def swc_to_line_actors(swc_df, axon_color='blue', dendrite_color='red', lw=2):
    '''
    Build vedo Lines actors for axon and dendrite segments directly
    from a parsed SWC dataframe. Bypasses morphapi/vedo tube merging.
    '''
    
    node_coords = swc_df.set_index('id')[['x', 'y', 'z']]

    # Only rows that have a valid parent
    has_parent = swc_df[swc_df['parent'] != -1]

    actors = []
    for neurite_type, color in [(2, axon_color), (3, dendrite_color)]:
        subset = has_parent[has_parent['type'] == neurite_type]
        if subset.empty:
            continue

        # Build (N, 3) start and end point arrays
        start_pts = node_coords.loc[subset['id']].values
        end_pts   = node_coords.loc[subset['parent']].values

        actor = Lines(start_pts, end_pts, c=color, lw=lw)
        actors.append(actor)

    return actors

def swap_for_brainrender(swcpath):
    """
    swap coordinates of an swc to be rendered with brainrender, when swc is acquired from the Allen Institute

    Parameters
    ----------
    swcpath : str
        path to the swc you need coordinates swapped for.

    Returns
    -------
    cell_actors : list
        a list of brainrender Line actors to be added to a Scene.

    """

    
    swc_df = pd.read_csv(
         swcpath,
         comment='#', 
         sep=r'\s+',
         names=['id', 'type', 'x', 'y', 'z', 'r', 'parent']
    )
    #need to swap x and z coordinates, brainrender swapped their axes so using swcs as we get them will render them rotated 90 deg
    x = swc_df['x']
    z = swc_df['z']
    swc_df['x'] = z
    swc_df['z'] = x
    cell_actors = swc_to_line_actors(swc_df, axon_color='green', dendrite_color='black', lw=2)
    return cell_actors
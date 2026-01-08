import statistics
import pandas as pd 
import os

folder = r"Z:/personal_data/Filip_Liebsch/SP8_Biocenter_small/2025-04-23_floxed/G134R_HET_project/data/"

def execute(root, tbl):
    print(f"Found table in: {root}")
    table_path = os.path.join(root, tbl)
    df = pd.read_csv(table_path, sep="\t")

    #2520 pixels 144.78 µm --> 0.057452380952 µm --> 0.003300776071
    df['ClusterSize'] = df['mEGFP-Gphn_area'] * 0.003300776071
    df['Cluster'] = df['ClusterSize'].between(0.1, 1)
    # synaptic clusters inside cell
    cell_syn_cluster_cols = ['vGAT', 'Neuron', 'Cluster', 'mScarlet-Gphn']
    cell_syn_mask = df[cell_syn_cluster_cols].all(axis=1)
    cell_syn_clusters = cell_syn_mask.sum()
    cell_syn_density = cell_syn_clusters/(df['Neuron_area'].iloc[0] * 0.003300776071)

    # synaptic clusters inside soma
    soma_syn_cluster_cols = ['vGAT', 'Soma', 'Cluster', 'mScarlet-Gphn']
    soma_syn_mask = df[soma_syn_cluster_cols].all(axis=1)
    soma_syn_clusters = soma_syn_mask.sum()
    soma_syn_density = soma_syn_clusters/(df['Soma_area'].iloc[0] * 0.003300776071)

    # synaptic cluster ratio
    cell_cluster_cols = ['Neuron', 'Cluster']
    cell_cluster_mask = df[cell_cluster_cols].all(axis=1)
    cell_clusters = cell_cluster_mask.sum()
    cell_synaptic_ratio = cell_syn_clusters/cell_clusters


    # area
    mean_area = df.loc[cell_syn_mask, 'ClusterSize'].mean()

    # intensity
    mean_intensity = df.loc[cell_syn_mask, 'mEGFP-Gphn_mean'].mean()

    # cell parameters
    soma_area = df['Soma_area'].iloc[0] * 0.003300776071
    neuron_area = df['Neuron_area'].iloc[0] * 0.003300776071

    # cell intensity parameters
    neuron_mEGFP_mean = df['Neuron_mEGFP-Gphn_mean'].iloc[0]
    soma_mEGFP_mean = df['Soma_mEGFP-Gphn_mean'].iloc[0]
    neuron_mScarlet_mean = df['Neuron_mScarlet-Gphn_mean'].iloc[0]
    soma_mScarlet_mean = df['Soma_mScarlet-Gphn_mean'].iloc[0]

    # cell info
    date = df['date'].iloc[0]
    condition = df['condition'].iloc[0]
    cell = df['cell'].iloc[0]

    # information summary dataframe
    statistics = {
        'synaptic_cell_density':        [cell_syn_density],
        'synaptic_soma_density':        [soma_syn_density],
        'synaptic_ratio':               [cell_synaptic_ratio],
        'mean_synaptic_area':           [mean_area],
        'mean_synaptic_intensity':      [mean_intensity],
        'soma_area':                    [soma_area],
        'neuron_area':                  [neuron_area],
        'neuron_mEGFP_mean':            [neuron_mEGFP_mean],
        'soma_mEGFP_mean':              [soma_mEGFP_mean],
        'neuron_mScarlet_mean':         [neuron_mScarlet_mean], 
        'soma_mScarlet_mean':           [soma_mScarlet_mean],
        'date':                         [date],
        'condition':                    [condition],    
        'cell':                         [cell]
    }


    df_summary = pd.DataFrame(statistics)
   

    path_norm = os.path.normpath(root)
    path_list = path_norm.split(os.sep)

    df_summary.to_csv(os.path.join(root, path_list[-5] + "_cluster_summary_" + path_list[-4] + "_" + path_list[-3] + ".tsv"), sep="\t")


for root, dirs, files in os.walk(folder):
    for file in files:
        if file.lower().endswith(".tsv") and "cluster_analysis" in file.lower():
            execute(root, file)

print("...completed.")
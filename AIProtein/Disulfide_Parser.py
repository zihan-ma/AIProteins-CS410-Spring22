from dis import dis
import geometry 
import parsers 
import util 
import torch
import os
import numpy as np

def disulfide_parser():
    # can convert this into a function when the time comes
    # Get the list of all files and directories
    pdb_list = np.empty((0, 3, 3))
    cys_pdb_list = np.empty((0, 3, 3))

    # Make sure to change directory to where PDBs are
    path = "C:\\Users\\name\\AIProteins-CS410-Spring22\\PDBs"
    dir_list = os.listdir(path)
    for x in dir_list:
        if x.endswith(".pdb"):
            new_f = path + "\\" + x
            pdb = parsers.parse_pdb(new_f)
            pdb_list = np.concatenate((pdb_list, pdb['xyz'][:,:3,:])) 
            cys_pdb_list = np.concatenate((pdb_list, pdb['cys_xyz'][:,:3,:]))

    xyz_ref = torch.tensor(pdb_list).float()
    cys_xyz_ref = torch.tensor(cys_pdb_list).float()

    # get ref dists
    c6d_ref = geometry.xyz_to_c6d(xyz_ref[None].permute(0,2,1,3),{'DMAX':20.0})
    c6d_ref2 = geometry.xyz_to_c6d(xyz_ref[None],{'DMAX':20.0})
    cys_c6d_ref = geometry.xyz_to_c6d(cys_xyz_ref[None].permute(0,2,1,3),{'DMAX':20.0})
    cys_c6d_ref2 = geometry.xyz_to_c6d(cys_xyz_ref[None],{'DMAX':20.0})
    # 0 is not cys bond, 1 is cys bond
    return (0, c6d_ref.numpy()[0]), (1, cys_c6d_ref.numpy()[0])


"""
Convert numpy format from (x, x, 4) to (y, 4) for neural network
"""
def adapter_utility(features: np.ndarray, label: int, dims: tuple) -> np.ndarray:
    h, w, d = dims
    listed = []
    for i in range(h):
        for j in range(w):
            buffer = []
            for k in range(d):
                buffer.append(features[i, j, k])
            buffer.append(label)
            listed.append(buffer)
    
    return np.array(listed)

def converter_utility(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    features = np.append(a1, a2, axis=0)

    np.random.shuffle(features)
    np.random.shuffle(features)
    
    return features

def adapter() -> np.ndarray:
    data = disulfide_parser()

    # get labels
    label0 = data[0][0]
    label1 = data[1][0]
    
    # get features
    raw_features0 = data[0][1]
    raw_features1 = data[1][1]
    
    # convert feature shapes
    features_and_labels0 = adapter_utility(raw_features0, label0, raw_features0.shape)
    features_and_labels1 = adapter_utility(raw_features1, label1, raw_features1.shape)

    # merge and shuffel
    features_and_labels = converter_utility(features_and_labels0, features_and_labels1)

    # [feature0, feature1, feature2, feature3, label]
    return features_and_labels
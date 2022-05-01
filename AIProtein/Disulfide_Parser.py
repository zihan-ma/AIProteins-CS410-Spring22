import geometry 
import parsers 
import util 
import torch
import os
import numpy as np
from dis import dis

"""
Should have path variable to parameter but left blank for now to avoid messing up stuff running on your end
add path to parameter when you make the proper changes
"""
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
    # c6d_ref2 = geometry.xyz_to_c6d(xyz_ref[None],{'DMAX':20.0})
    cys_c6d_ref = geometry.xyz_to_c6d(cys_xyz_ref[None].permute(0,2,1,3),{'DMAX':20.0})
    # cys_c6d_ref2 = geometry.xyz_to_c6d(cys_xyz_ref[None],{'DMAX':20.0})
    
    # 0 is not cys bond, 1 is cys bond
    return (0, c6d_ref.numpy()[0]), (1, cys_c6d_ref.numpy()[0])

# Same parser as above but does percentage parsing
# pct is a float from 0 to 1.0

def disulfide_parser_pct(path, count, pct):
    # can convert this into a function when the time comes
    # Get the list of all files and directories
    pdb_list = np.empty((0, 3, 3))
    cys_pdb_list = np.empty((0, 3, 3))
    total_to_parse = np.floor(count * pct)
    counter = 0
    
    
    # Make sure to change directory to where PDBs are
    dir_list = os.listdir(path)
    for x in dir_list:
        if counter == total_to_parse:
            break
        if x.endswith(".pdb"):
            new_f = path + "\\" + x
            pdb = parsers.parse_pdb(new_f)
            pdb_list = np.concatenate((pdb_list, pdb['xyz'][:,:3,:])) 
            cys_pdb_list = np.concatenate((pdb_list, pdb['cys_xyz'][:,:3,:]))
            counter += 1
        

    xyz_ref = torch.tensor(pdb_list).float()
    cys_xyz_ref = torch.tensor(cys_pdb_list).float() 
    # get ref dists
    c6d_ref = geometry.xyz_to_c6d(xyz_ref[None].permute(0,2,1,3),{'DMAX':20.0})
    cys_c6d_ref = geometry.xyz_to_c6d(cys_xyz_ref[None].permute(0,2,1,3),{'DMAX':20.0})
    
    # 0 is not cys bond, 1 is cys bond
    return (0, c6d_ref.numpy()[0]), (1, cys_c6d_ref.numpy()[0])
    
# Count total files in pdb database
def file_count(path):
    count = 0
    dir_list = os.listdir(path)
    for x in dir_list:
        if x.endswith(".pdb"):
            count += 1
    return count

# Used to parse a specific file
def specific_parse_file(path, file_name):
    pdb_list = np.empty((0, 3, 3))
    cys_pdb_list = np.empty((0, 3, 3))
    new_f = path + "\\" + file_name
    pdb = parsers.parse_pdb(new_f)
    pdb_list = np.concatenate((pdb_list, pdb['xyz'][:,:3,:])) 
    cys_pdb_list = np.concatenate((pdb_list, pdb['cys_xyz'][:,:3,:]))
    
    xyz_ref = torch.tensor(pdb_list).float()
    cys_xyz_ref = torch.tensor(cys_pdb_list).float() 
    # get ref dists
    c6d_ref = geometry.xyz_to_c6d(xyz_ref[None].permute(0,2,1,3),{'DMAX':20.0})
    cys_c6d_ref = geometry.xyz_to_c6d(cys_xyz_ref[None].permute(0,2,1,3),{'DMAX':20.0})
    
    # 0 is not cys bond, 1 is cys bond
    return (0, c6d_ref.numpy()[0]), (1, cys_c6d_ref.numpy()[0])
    
# Used to update the pdb_set   
def update_parser_db(parsed_pdb1, parsed_pdb2):
    update_1 = (0, np.concatenate((parsed_pdb1[0][1], parsed_pdb2[0][1]), axis=0))
    update_2 = (1, np.concatenate((parsed_pdb1[1][1], parsed_pdb2[1][1]), axis=0))
    return update_1, update_2

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



"""
    keep the numpy array format but add an additional channel for the label (x, x, 5)
    will make experimental trials with a 3D Convolution Volume.

"""
def callParser() -> tuple:

    data = disulfide_parser()

    # get features
    raw_features0 = data[0][1]
    raw_features1 = data[1][1]

    labeled_data = addLabels(raw_features0, raw_features1)
    nocys = labeled_data[0]
    cys   = labeled_data[1]
    shuffled_data = utility_shuffel3D(nocys, cys)
    nocys_base = shuffled_data[0]
    cys_base   = shuffled_data[1]
    return (nocys_base, cys_base)


def addLabels(nocys: np.ndarray, cys: np.ndarray) -> tuple:

    h0, w0, d0 = nocys.shape
    h1, w1, d1 = cys.shape
    nocys = np.concatenate((nocys, np.zeros((h0, w0, 1))), axis=2)
    cys   = np.concatenate((cys, np.ones((h1, w1, 1))), axis=2)


    return (nocys, cys)

def utility_shuffel3D(nocys: np.ndarray, cys: np.ndarray):
    h0, w0, d0 = nocys.shape
    h1, w1, d1 = cys.shape
    if h0 != w0 or h1 != w1:
        print("Error: 2D array shape does not match")
        return
    
    # basis for shuffel
    checker = 0
    h, w = 0, 0
    if h0 > h1:
        checker = 0 # 0 == using nocys
        h = h0
        w = w0
    elif h0 < h1:
        checker = 1 # 1 == using cys
        h = h1
        w = w1
    
    for i in range(h):
        for j in range(w):
            if (checker == 0 and (i < h1 and j < w1)) or (checker == 1 and (i < h0 and j < w0)):
                if np.random.uniform() > 0.5:

                    x = np.copy(nocys[i,j,:])
                    nocys[i,j,:] = cys[i,j,:]
                    cys[i,j,:] = x



    return (nocys, cys)
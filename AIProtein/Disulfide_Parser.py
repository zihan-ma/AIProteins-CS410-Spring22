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
    path = "C:\\Users\\wwe61\\Downloads\\PDBs"
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


     
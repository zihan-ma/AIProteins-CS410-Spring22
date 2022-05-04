import parsers
import geometry
import numpy as np

def parse(filename):
    pdb = parsers.parse_pdb(filename)
    shape = pdb['idx'].shape[0]
    for cys in pdb['ssbond']:
        #(cat1, index1, cat2, index2, dist)
        index1 = np.where(pdb['idx'] == cys[1])[0]
        index2 = np.where(pdb['idx'] == cys[3])[0]
        for i in index1:
            for j in index2:
                n1 = i*shape+j
                n2 = j*shape+i
                pdb['c6d'][n1][4] = 1
                pdb['c6d'][n2][4] = 1
    return pdb['c6d']

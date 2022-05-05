import Parser.Util.parser as parser
import numpy as np

def parse(filename):
    pdb = parser.parse_pdb(filename)
    if pdb == []:
        return np.array([])
    shape = pdb['idx'].shape[0]
    # has_ssbond = 0
    for cys in pdb['ssbond']:
        # cys shape: (cat1, index1, cat2, index2, dist)
        # has_ssbond = 1
        index1 = np.where(pdb['idx'] == cys[1])[0]
        index2 = np.where(pdb['idx'] == cys[3])[0]
        for i in index1:
            for j in index2:
                n1 = i*shape+j
                n2 = j*shape+i
                pdb['c6d'][n1][4] = 1
                pdb['c6d'][n2][4] = 1
    # return (has_ssbond, pdb['c6d'])
    return pdb['c6d']
# code modified from https://github.com/RosettaCommons/RFDesign/tree/main/hallucination/util/geometry.py

import numpy as np
import torch

# ============================================================
def get_pair_dist(a, b):
    """calculate pair distances between two sets of points

    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """

    dist = torch.cdist(a, b, p=2)
    return dist

# ============================================================
def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    v = v / torch.norm(v, dim=-1, keepdim=True)
    w = w / torch.norm(w, dim=-1, keepdim=True)

    # this is not stable at the poles
    #vw = torch.sum(v*w, dim=-1)
    #ang = torch.acos(vw)

    # this is better
    # https://math.stackexchange.com/questions/1143354/numerically-stable-method-for-angle-between-3d-vectors/1782769
    y = torch.norm(v-w,dim=-1)
    x = torch.norm(v+w,dim=-1)
    ang = 2*torch.atan2(y, x)

    return ang

# ============================================================
def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor of shape [batch,nres]
          stores resulting dihedrals
    """
    b0  = a - b
    b1r = c - b
    b2  = d - c

    b1 = b1r/torch.norm(b1r, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1

    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)
    ang = torch.atan2(y, x)

    return ang


# ============================================================
def xyz_to_c6d(xyz, params):
    """convert cartesian coordinates into 2d distance
    and orientation maps

    Parameters
    ----------
    xyz : pytorch tensor of shape [batch,3,nres,3]
          stores Cartesian coordinates of backbone N,Ca,C atoms
    Returns
    -------
    c6d : pytorch tensor of shape [batch,nres,nres,4]
          stores stacked dist,omega,theta,phi 2D maps
    """

    batch = xyz.shape[0]
    nres = xyz.shape[2]

    # three anchor atoms
    N  = xyz[:,0]
    Ca = xyz[:,1]
    C  = xyz[:,2]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    # 6d coordinates order: (dist,omega,theta,phi)
    c6d = torch.zeros([batch,nres,nres,4],dtype=xyz.dtype,device=xyz.device)

    dist = get_pair_dist(Cb,Cb)
    dist[torch.isnan(dist)] = 999.9
    c6d[...,0] = dist + 999.9*torch.eye(nres,device=xyz.device)[None,...]
    b,i,j = torch.where(c6d[...,0]<params['DMAX'])

    c6d[b,i,j,torch.full_like(b,1)] = get_dih(Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j])
    c6d[b,i,j,torch.full_like(b,2)] = get_dih(N[b,i], Ca[b,i], Cb[b,i], Cb[b,j])
    c6d[b,i,j,torch.full_like(b,3)] = get_ang(Ca[b,i], Cb[b,i], Cb[b,j])

    # fix long-range distances
    c6d[...,0][c6d[...,0]>=params['DMAX']] = 999.9

    return c6d

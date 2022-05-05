#!/bin/bash
RCSB_PDB="ftp://ftp.wwpdb.org/pub/pdb/data/structures/all/pdb/*"
TARGET="$PWD/Data/raw/"
wget -N -P $TARGET $RCSB_PDB
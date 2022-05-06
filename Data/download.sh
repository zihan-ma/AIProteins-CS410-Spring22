#!/bin/bash
RCSB_PDB="ftp://ftp.wwpdb.org/pub/pdb/data/structures/all/pdb/*"
TARGET="$PWD/Data/Raw/"
wget -N -P $TARGET $RCSB_PDB
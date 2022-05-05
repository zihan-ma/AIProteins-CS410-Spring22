import subprocess
import sys
import os
import argparse
import Parser.parse_pdb as parser
import numpy as np
import gzip

# directories
cwd = os.getcwd()
raw_fp = "/Data/Raw/"
pdb_fp = "/Data/PDB/"
parsed_fp = "/Data/Parsed/"
zip_ext = ".ent.gz"
pdb_ext = ".pdb"
parse_ext = ".pars"
# has_ss = "has-ssbond/"
# no_ss = "no-ssbond/"

# parsing command line arguments
argp = argparse.ArgumentParser()
argp.add_argument("-d", "--download", action="store_true", help="check the PDB for updates, or download the PDB; zipped files are stored in Data/raw")
argp.add_argument("-u", "--unzip", action="store_true", help="unzip the compressed downloaded PDB files; unzipped files are stored in Data/pdb")
# argp.add_argument("-p", "--parse", action="store_true", help="parse the pdb files and categorize them based on ssbonds; output files are stored in Data/parsed")
argp.add_argument("-p", "--parse", action="store_true", help="parse the pdb files; output files are stored in Data/parsed")
argp.add_argument("-s", "--silent", action="store_true", help="suppress runtime information")
args = argp.parse_args()

# running
if not (args.download or args.unzip or args.parse):
    argp.print_help()
    exit(0)
if args.download:
    os.makedirs(os.path.dirname(cwd + raw_fp), exist_ok=True)
    if args.silent:
        proc = subprocess.Popen('/bin/bash', text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
    else:
        proc = subprocess.Popen('/bin/bash', text=True, stdin=subprocess.PIPE, stdout=sys.stdout, stderr=sys.stderr)
    proc.communicate('Data/download.sh')
if args.unzip:
    os.makedirs(os.path.dirname(cwd + pdb_fp), exist_ok=True)
    zipped = os.listdir(cwd + raw_fp)
    unzipped = os.listdir(cwd + pdb_fp)
    for pdb in zipped:
        if not pdb.endswith(zip_ext):
            continue
        name = pdb[:pdb.find(zip_ext)]
        fullname = name + pdb_ext
        rawpath = cwd + raw_fp + pdb
        pdbpath = cwd + pdb_fp + fullname
        if not (fullname in unzipped and os.path.getmtime(rawpath) < os.path.getmtime(pdbpath)):
            with gzip.open(rawpath, "rt") as infile:
                content = infile.read()
            if not args.silent:
                print("unzipping", name)
            with open(pdbpath, "w") as outfile:
                outfile.write(content)
        else:
            if not args.silent:
                print(name,"up to date")
if args.parse:
    os.makedirs(os.path.dirname(cwd + parsed_fp), exist_ok=True)
    zipped = os.listdir(cwd + raw_fp)
    unzipped = os.listdir(cwd + pdb_fp)
    parsed = os.listdir(cwd + parsed_fp)
    for pdb in unzipped:
        if not pdb.endswith(pdb_ext):
            continue
        name = pdb[:pdb.find(pdb_ext)]
        fullname = name + parse_ext
        rawpath = cwd + raw_fp + name + zip_ext
        pdbpath = cwd + pdb_fp + pdb
        parsedpath = cwd + parsed_fp + fullname
        if not (fullname in parsed and os.path.getmtime(rawpath) < os.path.getmtime(parsedpath)):
            data = parser.parse(pdbpath)
            if not np.array_equiv(np.array([]), data):
                if not args.silent:
                    print(name, "parsed")
                np.savetxt(parsedpath, data)
            else:
                if not args.silent:
                    print(name,"has no relevant residues")
        else:
            if not args.silent:
                print(name,"already parsed")
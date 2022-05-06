import numpy as np
import subprocess
import sys
import os
import argparse
import gzip
import shutil
import Parser.parse_pdb as parser

# directories
cwd = os.getcwd()
raw_fp = "/Data/Raw/"
pdb_fp = "/Data/PDB/"
parsed_fp = "/Data/Parsed/"
have_ss_fp = "/Data/Have_SS/"
no_ss_fp = "/Data/No_SS/"
zip_ext = ".ent.gz"
pdb_ext = ".pdb"
parse_ext = ".csv"

# parsing command line arguments
argp = argparse.ArgumentParser()
argp.add_argument("-d", "--download", action="store_true", help="check the PDB for updates, or download the PDB; zipped files are stored in Data/raw")
argp.add_argument("-u", "--unzip", action="store_true", help="unzip the compressed downloaded PDB files; unzipped files are stored in Data/pdb")
argp.add_argument("-p", "--parse", action="store_true", help="parse the pdb files; output files are stored in Data/parsed")
argp.add_argument("-o", "--organize", action="store_true", help="sort parsed pdb files on existence of a disulfide bonds")
argp.add_argument("-s", "--silent", action="store_true", help="suppress runtime information")
args = argp.parse_args()

# running
if not (args.download or args.unzip or args.parse or args.organize):
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
        raw_path = cwd + raw_fp + pdb
        pdb_path = cwd + pdb_fp + fullname
        if not (fullname in unzipped and os.path.getmtime(raw_path) < os.path.getmtime(pdb_path)):
            with gzip.open(raw_path, "rt") as infile:
                content = infile.read()
            if not args.silent:
                print("unzipping", name)
            with open(pdb_path, "w") as outfile:
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
        raw_path = cwd + raw_fp + name + zip_ext
        pdb_path = cwd + pdb_fp + pdb
        parsed_path = cwd + parsed_fp + fullname
        if not (fullname in parsed and os.path.getmtime(raw_path) < os.path.getmtime(parsed_path)):
            if not args.silent:
                print(name,end=" ")
            data = parser.parse(pdb_path)
            if np.array_equiv(np.array([1]), data) and not args.silent:
                print("parse failed")
            elif np.array_equiv(np.array([2]), data) and not args.silent:
                print("has no CYS residues")
            else:
                if not args.silent:
                    print("parse successful")
                np.savetxt(parsed_path, data, fmt=['%f','%f','%f','%f','%d','%s','%d','%s','%d'], delimiter=',')
        else:
            if not args.silent:
                print(name, "already parsed")
if args.organize:
    os.makedirs(os.path.dirname(cwd + have_ss_fp), exist_ok=True)
    os.makedirs(os.path.dirname(cwd + no_ss_fp), exist_ok=True)
    parsed = os.listdir(cwd + parsed_fp)
    have_ss = os.listdir(cwd + have_ss_fp)
    no_ss = os.listdir(cwd + no_ss_fp)
    for pdb in parsed:
        if not pdb.endswith(parse_ext):
            continue
        name = pdb[:pdb.find(parse_ext)]
        parsed_path = cwd + parsed_fp + pdb
        have_ss_path = cwd + have_ss_fp + pdb
        no_ss_path = cwd + no_ss_fp + pdb
        pdb_data = np.loadtxt(parsed_path, dtype=parser.type, delimiter=',')
        has_ss = False
        if pdb_data.shape == ():
            pdb_data = np.array([pdb_data])
        for line in pdb_data:
            if line[4] == 0:
                has_ss = True
                break
        if has_ss:
            if not args.silent:
                print(name, "has a disulfide bond")
            shutil.copyfile(parsed_path, have_ss_path)
        else:
            if not args.silent:
                print(name, "has no disulfide bond")
            shutil.copyfile(parsed_path, no_ss_path)
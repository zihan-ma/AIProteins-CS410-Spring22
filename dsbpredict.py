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
rich_ss_fp = "/Data/Rich_SS/"
sparse_ss_fp = "/Data/Sparse_SS/"
no_ss_fp = "/Data/No_SS/"
zip_ext = ".ent.gz"
pdb_ext = ".pdb"
parse_ext = ".csv"

# parsing command line arguments
argp = argparse.ArgumentParser()
argp.add_argument("-d", "--download", action="store_true", help="check the PDB for updates, or download the PDB; zipped files are stored in Data/raw")
argp.add_argument("-u", "--unzip", action="store_true", help="unzip the compressed downloaded PDB files; unzipped files are stored in Data/pdb")
argp.add_argument("-p", "--parse", action="store_true", help="parse the pdb files; output files are stored in Data/parsed")
argp.add_argument("-o", "--organize", action="store_true", help="sort parsed pdb files on disulfide bonds")
argp.add_argument("-s", "--silent", action="store_true", help="suppress runtime information")
args = argp.parse_args()

# running
if not (args.download or args.unzip or args.parse or args.organize):
    argp.print_help()
    exit(0)
else:
    os.makedirs(os.path.dirname(cwd + raw_fp), exist_ok=True)
    os.makedirs(os.path.dirname(cwd + pdb_fp), exist_ok=True)
    os.makedirs(os.path.dirname(cwd + parsed_fp), exist_ok=True)
    os.makedirs(os.path.dirname(cwd + rich_ss_fp), exist_ok=True)
    os.makedirs(os.path.dirname(cwd + sparse_ss_fp), exist_ok=True)
    os.makedirs(os.path.dirname(cwd + no_ss_fp), exist_ok=True)
    
if args.download:
    proc = subprocess.Popen('/bin/bash', text=True, stdin=subprocess.PIPE, stdout=sys.stdout, stderr=sys.stderr)
    proc.communicate('Data/download.sh')
if args.unzip:
    zipped = os.listdir(cwd + raw_fp)
    zipped.sort()
    unzipped = os.listdir(cwd + pdb_fp)
    unzipped.sort()
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
                print(name, "unzipping")
            with open(pdb_path, "w") as outfile:
                outfile.write(content)
        else:
            if not args.silent:
                print(name, "up to date")
if args.parse:
    zipped = os.listdir(cwd + raw_fp)
    zipped.sort()
    unzipped = os.listdir(cwd + pdb_fp)
    unzipped.sort()
    parsed = os.listdir(cwd + parsed_fp)
    parsed.sort()
    failed_fp = cwd + "/Data/failed.csv"
    open(failed_fp, "a")
    with open(failed_fp, "r+") as f:
        failed = f.readlines()
        for pdb in unzipped:
            name = pdb[:pdb.find(pdb_ext)]
            fullname = name + parse_ext
            raw_path = cwd + raw_fp + name + zip_ext
            pdb_path = cwd + pdb_fp + pdb
            parsed_path = cwd + parsed_fp + fullname
            if not (fullname in parsed and os.path.getmtime(raw_path) < os.path.getmtime(parsed_path)) and name+'\n' not in failed:
                if not args.silent:
                    print(name, end=" ")
                errc, data = parser.parse(pdb_path)
                if errc == 1 and not args.silent:
                    print("parse failed")
                    f.write(name + '\n')
                elif errc == 2 and not args.silent:
                    print("has no CYS residues")
                    f.write(name + '\n')
                elif errc == 3 and not args.silent:
                    print("has no bondable CYS residues")
                    f.write(name + '\n')
                else:
                    if not args.silent:
                        print("parse successful")
                    np.savetxt(parsed_path, data, fmt=parser.csv_format, delimiter=',')
            else:
                if not args.silent:
                    print(name, "already parsed")
if args.organize:
    parsed = os.listdir(cwd + parsed_fp)
    parsed.sort()
    rich_ss = os.listdir(cwd + rich_ss_fp)
    rich_ss.sort()
    sparse_ss = os.listdir(cwd + sparse_ss_fp)
    sparse_ss.sort()
    no_ss = os.listdir(cwd + no_ss_fp)
    no_ss.sort()
    for pdb in parsed:
        name = pdb[:pdb.find(parse_ext)]
        raw_path = cwd + raw_fp + name + zip_ext
        parsed_path = cwd + parsed_fp + pdb
        rich_ss_path = cwd + rich_ss_fp + pdb
        sparse_ss_path = cwd + sparse_ss_fp + pdb
        no_ss_path = cwd + no_ss_fp + pdb
        if not args.silent:
            print(name, end=" ")
        if not (pdb in (rich_ss + sparse_ss + no_ss) and os.path.getmtime(raw_path) < os.path.getmtime(parsed_path)):
            pdb_data = np.loadtxt(parsed_path, dtype=parser.csv_type, delimiter=',')
            ss = 0
            nss = 0
            if pdb_data.shape == ():
                pdb_data = np.array([pdb_data])
            for line in pdb_data:
                if line[4] == 1:
                    ss += 1
                else:
                    nss += 1
            if ss == 0:
                if not args.silent:
                    print("has no disulfide bond")
                shutil.copyfile(parsed_path, no_ss_path)
            else:
                if 10*ss >= nss:
                    if not args.silent:
                        print("has high disulfide density")
                    shutil.copyfile(parsed_path, rich_ss_path)
                else:
                    if not args.silent:
                        print("has low disulfide density")
                    shutil.copyfile(parsed_path, sparse_ss_path)
        else:
            if not args.silent:
                print("already sorted")
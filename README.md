# AI Proteins
Software version 1.0.0

## Running required Modules:

Anaconda verson == xxx  
Python verson == xxx  
Jupyter Notebook == xxx  
xxx verson == xxx  
xxx verson == xxx  

## Chimera script

#### Allocate resource on chimera  

Try a different partition or specify multiple partitions (comma separated list) to allow it to use any partition on the list you give it.  

List for the cores: https://www.umb.edu/rc/hpc/chimera/chimera_scheduler  

$ srun -n 4 -N 1 -p Intel6126,Intel6240,AMD6276 -t 01:00:00 --pty /bin/bash  

#### Check the current waiting list  

$ squeue  

#### Help info  

$ sinfo  













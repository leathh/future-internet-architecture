# Train VGG-19 on Heterogeneous Servers Using Horovod

Here you can find the project code of the assignment for Future Internet Architecture.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### 1. Create Conda environment

Create conda environment on every server using the following commands.

```
conda create -n horovod pip python=3.7
conda activate horovod
pip install --upgrade tensorflow
conda install -c conda-forge openmpi
conda install gxx_linux-64
pip install horovod
```

### 2. Create passwordless SSH connection

Make a SSH connection without password between all the servers respectively following the following tutorial.

```
https://www.tecmint.com/ssh-passwordless-login-using-ssh-keygen-in-5-easy-steps/
```

### 3. Add “conda activate horovod” to .bashrc

### 4. Clone git repository

```
https://github.com/leathh/future-internet-architecture.git
```

### 5. Edit run.sh and run-with-logging.sh

In the run.sh file, change all occurences of USERNAME in filepaths to your own username, e.g.:

```
/home/USERNAME/.conda/envs/horovod/bin/mpirun
```
change this and save it.


## Running it

### Simple execution

On server 1 (nasp-cpu-01), execute the run file. 
This will create a log directory with checkpoints stored and Tensorboard logging.

```
. run.sh
```

### Execution with more detailed logging data

On server 1 (nasp-cpu-01), execute the run file. 
This will create a log directory with checkpoints stored and Tensorboard logging.
Additionally, it will print out time stamps in the terminal, which will be stored in a stdout.log and stderr.log file.

```
. run-with-logging.sh
```


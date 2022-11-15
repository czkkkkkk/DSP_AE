# This is the experiment code of DSP for the AE process of PPoPP23
## Setup environment
1. Install docker on a GPU server and add NVIDIA Runtime for the docker (User guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html).
2. Download the docker from our docker hub repository using 'docker pull zhouqihui/dsp-ppopp-ae:latest'.
3. Run the docker with 'docker run -it --gpus all zhouqihui/dsp-ppopp-ae:latest /bin/bash'.
4. Pull the github repo using 'git clone https://github.com/czkkkkkk/DSP_AE.git'

## Run experiments
### Prepare data
The dataset preprocessing step uses different scripts for DSP and baselines.

Since the experiments of DSP are on partitioned graph, we partition the datasets and store them on the disk. We use the script partition.sh to download and partition a graph. Users only need to provide the graph name and the number of partitions. For example, "./partition.sh products 4" partitions Products into 4 parts. The partitioned graph are stored under "/data/ds/", which is used as default data directory in the later experiments.
    
We use a script preprocess.sh to download all datasets and convert them into the formats required by DGL, PyG and Quiver. The processed datasets for three baseline systems are stored under "/data/dgl/", "/data/pyg/", and "/data/quiver/", respectively.

### Run sampling experiment
bash sample.sh

### Run end-to-end experiment
bash train.sh

## Reproducing Tactical Optimistic and Pessimistic estimation (TOP) paper

Reproducing the results shown in the Figure 2,3 of the paper [Tactical Optimism and Pessimism for Deep Reinforcement Learning](https://arxiv.org/abs/2102.03765). The tasks are HalfCheetah-v2 and Hopper-v2 from Mujoco simulation library. 


### Installation:
--- 

First, log in to the desired GPU node and create the conda environment:

```bash
12345678@ppti-gpu-3 $ conda env create -f conda_environment.yaml
12345678@ppti-gpu-3 $ source conda.sh
12345678@ppti-gpu-3 $ conda activate TOP_Mujoco
```


Then clone the github repo and cd to the mujoco tasks folder:

```bash
(TOP_Mujoco) 12345678@ppti-gpu-3 $ git clone https://github.com/samasat5/TOP_Reproduce.git
(TOP_Mujoco) 12345678@ppti-gpu-3 $ cd TOP/mujoco
```
Then run the ```train_top_agent.py``` file:

```bash
(TOP_Mujoco) 12345678@ppti-gpu-3:~/TOP/mujoco$ python train_top_agent.py --env HalfCheetah-v3 --seed 0 --fixed_beta 0 # for Optimistic run, for pessimistic run: --fixed_beta -1
```
 



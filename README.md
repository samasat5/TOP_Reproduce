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

When running MuJoCo tasks on a remote GPU node (without display), you must enable **EGL-based rendering** and make sure the system can find the MuJoCo and NVIDIA library paths.

Add the following lines once:

```bash
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$HOME/.mujoco/mujoco210/bin"
for p in /usr/lib/nvidia /usr/lib/nvidia-* /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/nvidia; do
  [ -d "$p" ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$p"
done
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda-12.9/lib64"
```



### Training:
---

Then clone the github repo and cd to the mujoco tasks folder:

```bash
(TOP_Mujoco) 12345678@ppti-gpu-3 $ git clone https://github.com/samasat5/TOP_Reproduce.git
(TOP_Mujoco) 12345678@ppti-gpu-3 $ cd TOP/mujoco
```


Then run the ```train_top_agent.py``` file:

```bash
(TOP_Mujoco) 12345678@ppti-gpu-3:~/TOP/mujoco$ python train_top_agent.py --env HalfCheetah-v3 --seed 0 --fixed_beta 0 # for Optimistic run, for pessimistic run: --fixed_beta -1
```
### Logging and Export:
---
 
Each training run writes TensorBoard logs under: ```dope_runs/DOPE_<ENV>_nq<N>_<bandit_lr>_seed<SEED>/```. Inside that folder you’ll see one or more TensorBoard event files named like: ```events.out.tfevents.<timestamp>.<host>.<pid>.0```

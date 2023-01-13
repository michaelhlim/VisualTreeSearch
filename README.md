# Compositional Learning-based Planning for Vision POMDPs
This is the codebase for Compositional Learning-based Planning for Vision POMDPs [[Deglurkar, Lim, et al.]](https://arxiv.org/abs/2112.09456).

## Visual Tree Search (VTS)
![Visual Tree Search](misc/visual_tree_search_final.png)

### Summary of VTS
In the Visual Tree Search (VTS) algorithm, we integrate the learned POMDP model components with
classical filtering and planning techniques. This results in a POMDP solver that can learn model
components that are more sample efficient and interpretable than end-to-end approaches. It also
benefits from having a robust particle filter and planner built upon techniques with theoretical guar-
antees, and can adapt to different task rewards. To integrate planning techniques that use online tree
search, we must have access to a conditional generative model that can generate image observations
o from a given state s according to the likelihood density Z(o|s).

# Setup
## Installing Packages via Conda
It's most preferred to make your own Conda environment for the project. In order to do so, you can perform the following steps. First, create your own conda environment. The environment I used currently has Python 3.8.8:

```
conda create -name vts python=3.8
source activate vts
```

Next, add the relevant packages. I did a `<pip freeze>` in the current repository to get the new up-to-date `<requirements.txt>`. First, try to see if this works.

```
conda install --file requirements.txt
```

If this works, great! If not, here is the way I went about it. Try to remove the environment and start fresh. First, I installed the older PyTorch:

```
# For Linux/Windows
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=9.2 -c pytorch

# For MacOS
conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
```

Then, I installed the rest of the packages frozen by Johnathan:

```
conda install --file misc/requirements.txt
```


## Running Experiments
In order to run the experiments, run the following command:

```
conda activate vts
python scripts/[experiment name].py
```

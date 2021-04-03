# Visual Tree Search
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
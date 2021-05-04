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


## Installing and Using Deepmind Lab

Create a conda environment with numpy:
```
conda create --name deepmind
conda activate deepmind
conda install numpy
```

Download the Deepmind Lab repo:

```
git clone https://github.com/deepmind/lab.git
cd lab
```

### Installing and Building Bazel

Make sure you install g++, unzip, and zip if you don't have it already:

```
sudo apt install g++ unzip zip
```

It's best to download the latest version of Bazel, which is 4.0.0:

```
wget https://github.com/bazelbuild/bazel/releases/download/4.0.0/bazel-4.0.0-installer-linux-x86_64.sh
```

Then install it with

```
chmod +x bazel-4.0.0-installer-linux-x86_64.sh
./bazel-4.0.0-installer-linux-x86_64.sh --user
```

Add the following to your `~/.bashrc`:

```
export PATH="$PATH:$HOME/bin"
```

Now look at the file called `WORKSPACE` inside `lab`. At the bottom you might see something like this:

```
new_local_repository(
    name = "python_system",
    build_file = "@//bazel:python.BUILD",
    path = "/usr/",
)
```

This is fine as is if you want to use the version of python installed systemwide. However, if you want to use the version of python in your virtual environment, change the `path` line to, for example,

```
path = "/home/sampada_deglurkar/anaconda3/envs/deepmind/"
```

Now go to the file called `bazel/python.BUILD` in `lab`. Assuming you are using some version of python3, you want to change the corresponding `PY3` lines in the first `cc_library` chunk in the file to point to where your python and numpy are installed. So for example my `bazel/python.BUILD` file looks like this:

```
# Description:
#   Build rule for Python and Numpy.
#   This rule works for Debian and Ubuntu, and for MacOS. Other platforms might
#   keep the headers in different places.

cc_library(
    name = "python_headers_linux",
    hdrs = select(
        {            
            "@bazel_tools//tools/python:PY2": glob(["include/python2.7/*.h", "local/lib/python2.7/dist-packages/numpy/core/include/**/*.h"]),                       
            "@bazel_tools//tools/python:PY3": glob(["include/python3.6m/*.h", "lib/python3.6/site-packages/numpy/core/include/**/*.h"]),
        },
        no_match_error = "Internal error, Python version should be one of PY2 or PY3",
    ),
    includes = select(
        {
            "@bazel_tools//tools/python:PY2": ["include/python2.7", "local/lib/python2.7/dist-packages/numpy/core/include"],
            "@bazel_tools//tools/python:PY3": ["include/python3.6m", "lib/python3.6/site-packages/numpy/core/include"]
        },
        no_match_error = "Internal error, Python version should be one of PY2 or PY3",
    ),
)

cc_library(
    name = "python_headers_macos",
    hdrs = select(
        {
            "@bazel_tools//tools/python:PY2": glob(["Library/Frameworks/Python.framework/Versions/2.7/Headers/*.h"]),
            "@bazel_tools//tools/python:PY3": glob(["Library/Frameworks/Python.framework/Versions/3.9/Headers/**/*.h"]),
        },
        no_match_error = "Internal error, Python version should be one of PY2 or PY3",
    ),
    includes = select(
        {
            "@bazel_tools//tools/python:PY2": ["Library/Frameworks/Python.framework/Versions/2.7/Headers"],
            "@bazel_tools//tools/python:PY3": ["Library/Frameworks/Python.framework/Versions/3.9/Headers"],
        },
        no_match_error = "Internal error, Python version should be one of PY2 or PY3",
    ),
)

alias(
    name = "python_headers",
    actual = select(
        {
            "@//:is_linux": ":python_headers_linux",
            "@//:is_macos": ":python_headers_macos",
        },
        no_match_error = "Unsupported platform; only Linux and MacOS are supported.",
    ),
    visibility = ["//visibility:public"],
)

alias(
    name = "python",
    actual = ":python_headers",
    visibility = ["//visibility:public"],
)
```

Paths in the build rules in this file are relative to the root path specified in the `WORKSPACE` file. So for me `"include/python3.6m/*.h"` means `/home/sampada_deglurkar/anaconda3/envs/deepmind/include/python3.6m/*.h`. The `"lib/python3.6/site-packages/numpy/core/include/"` part I got by running `import numpy as np` and then `np.get_include()`, which gave me `/home/sampada_deglurkar/anaconda3/envs/deepmind/lib/python3.6/site-packages/numpy/core/include`. 

Now you should be able to build Bazel:

```
bazel build -c opt //:deepmind_lab.so
```

Now run the following inside `lab`:

```
bazel build -c opt //python/pip_package:build_pip_package
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
```

The second command above creates a `.whl` file in `/tmp/dmlab_pkg`. Install it using:

```
pip install /tmp/dmlab_pkg/deepmind_lab-1.0-py3-none-any.whl
```
(That’s the file that was in my `/tmp/dmlab_pkg`.)


Then you might also need to run

```
pip install /tmp/dmlab_pkg/deepmind_lab-1.0-py3-none-any.whl[dmenv_module]
```


### Testing the Installation

Create a file `agent.py` with the following code:

```
import deepmind_lab
import numpy as np

# Create a new environment object.
lab = deepmind_lab.Lab("demos/extra_entities", ['RGB_INTERLEAVED'],
                       {'fps': '30', 'width': '80', 'height': '60'})
lab.reset(seed=1)

# Execute 100 walk-forward steps and sum the returned rewards from each step.
print(sum(
    [lab.step(np.array([0,0,0,1,0,0,0], dtype=np.intc)) for i in range(0, 100)]))
```

Then run it with `python agent.py`. It should print out a reward of 4.0.


I used this helpful link as a main reference: https://github.com/deepmind/lab/tree/master/python/pip_package







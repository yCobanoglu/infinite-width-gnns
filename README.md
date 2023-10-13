# Code for Infinite Width Graph Neural Networks for Node Regression/ Classification

## Paper
[Infinite Width Graph Neural Networks for Node Regression/ Classification] (https://arxiv.org/abs/2310.08176) 



## Requirements
- \>= Python 3.10
- Julia and Laplacians.jl for [Effective Resistance](https://github.com/danspielman/Laplacians.jl). Julia will get intsalled automatically by juliacall in requirements.txt (once installed you have to call the julia binary in your venv and from there install Laplacians: `import Pkg; Pkg.add("Laplacians")` 
- [Pytorch](https://pytorch.org/), [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), [Neural Tangents](https://github.com/google/neural-tangents)
- [Sparse-Dot-MKL](https://github.com/flatironinstitute/sparse_dot) which requires Intel MKL
- LD_LIBRARY_PATH needs to be set for lgnn/gnn which use https://github.com/facebookresearch/bitsandbytes
- Other dependencies can be installed with `gnn/requirements.txt`
- To run tests install `gnn/requirements-dev.txt` (uses pytest)
- To see an example installation for ubuntu, see installation.txt

## Running Experiments

### Graph Neural Networks
Run `gnn/run_experiments.py` with path to directory that contains a `config.yaml` file. <br/>
For example `gnn/run_experiments.py ./` will use the `gnn/config.yaml`. <br/>
For example configs see `gnn/example-configs/` directory. <br/>
`gnn/run_experiments.py ./ -t` will use the `gnn/config.yaml` and make a test run (meaning one epoch only to check if everything works). <br/>
`PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/opt/cuda/lib64;BITSANDBYTES_NOWELCOME=1;PYTHONOPTIMIZE=1 gnn/run_experiments.py ./` has to be used for running experiments in parallel <br/>
`PYTHONOPTIMIZE=1` will remove assert statements and has to be set to spawn processes within processes (which will been spawned by run_experiments.py)

### Gaussian Processes and Kernels
### For some quick runs 
Classification and Regression for all Models and multiple datasets with one command, but GAT models need to be run seperatly because it uses 0-1 Adjacency Matrix and other models use WellingNormalized Adjacency Matrix <br/>
Kernels and GP do not use cuda. <br/>
export MKL_DYNAMIC=FALSE <br/>
export MKL_INTERFACE_LAYER=ILP64 <br/>
ef stands for effective resistance (how many percent of edges should be deleted) <br/>

 - `python3 gnn/infinite_width/run.py --datasets "cora citeseer" --models "nn gnn sgnn"`
 - Large Datasets should be run per model (because for multiple models kernel is stored in memory)
 - `python3 gnn/infinite_width/run.py --datasets "pubmed" --models "nn"`
 - `python3 gnn/infinite_width/run.py --datasets "pubmed" --models "gnn"`
 - `python3 gnn/infinite_width/run.py --datasets "pubmed" --models "sgnn"`
 - `python3 gnn/infinite_width/run.py --datasets "cornell" --models "gat"`

### Reproducing Results
 - `python3 gnn/infinite_width/run.py --datasets "cora citeseer pubmed" --models "nn gnn sgnn"`
 - `python3 gnn/infinite_width/run.py --datasets "cora" --models "gat"`
 - `python3 gnn/infinite_width/run.py --datasets "citeseer" --models "gat"`
 - `python3 gnn/infinite_width/run.py --datasets "pubmed" --models "gat" --ef 0.9`
 - `python3 gnn/infinite_width/run.py --datasets "chameleon" --models "gat"`
 - `python3 gnn/infinite_width/run.py --datasets "squirrel" --models "gat" --ef 0.9`
 - `python3 gnn/infinite_width/run.py --datasets "crocodile" --models "gat" --ef 0.9`

## Simulations
 Figures in Simulations section created with `python3 gnn/figures/main.py`

## Note
- `torch.compile` makes everything slower and not working with sparse spmm


## Acknowledgements
- https://arxiv.org/abs/1609.02907
- https://people.eecs.berkeley.edu/~sojoudi/Robust_GCN.pdf
- https://openreview.net/pdf?id=H1g0Z3A9Fm
- https://docs.dgl.ai/en/latest/tutorials/models/1_gnn/6_line_graph.html
- https://arxiv.org/pdf/2010.10046.pdf
- https://graph-tool.skewed.de/static/doc/spectral.html#graph_tool.spectral.hashimoto

## Some parts of code from (noted in files):
- https://github.com/zhengdao-chen/GNN4CD
- https://github.com/tkipf/pygcn



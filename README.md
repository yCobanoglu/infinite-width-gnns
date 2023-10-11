# Graph Neural Networks

## Requirements
- \>= Python 3.10
- Julia and Laplacians.jl for [Effective Resistance](https://github.com/danspielman/Laplacians.jl)
- [Pytorch](https://pytorch.org/), [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), [Neural Tangents](https://github.com/google/neural-tangents)
- [Sparse-Dot-MKL](https://github.com/flatironinstitute/sparse_dot) which requires Intel MKL
- LD_LIBRARY_PATH needs to be set for lgnn/gnn which use https://github.com/facebookresearch/bitsandbytes
- Other dependencies can be installed with `gnn/requirements.txt`
- To run tests install `gnn/requirements-dev.txt` (uses pytest)

## Running Experiments

### Graph Neural Networks
Run `gnn/run_experiments.py` with path to directory that contains a `config.yaml` file. <br/>
For example `gnn/run_experiments.py ./` will use the `gnn/config.yaml`. <br/>
For example configs see `gnn/configs/` directory. <br/>
`gnn/run_experiments.py ./ -t` will use the `gnn/config.yaml` and make a test run (meaning one epoch only to check if everything works). <br/>
`PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/opt/cuda/lib64;BITSANDBYTES_NOWELCOME=1;PYTHONOPTIMIZE=1 gnn/run_experiments.py ./` has to be used for running experiments in parallel <br/>
`PYTHONOPTIMIZE=1` will remove assert statements and has to be set to spawn processes within processes (which will been spawned by run_experiments.py)

### Gaussian Processes and Kernels
### For some quick runs 
Classification and Regression for all Models and multiple datasets per command, but GAT needs extra command because it uses 0-1 Adjacency Matrix and other models use WellingNormalized Adjacency
 - MKL_INTERFACE_LAYER=ILP64 python3 gnn/infinite_width/run.py --datasets "cora citeseer" --models "nn gnn sgnn"
 - Large Datasets should be run per model (because for multiple models kernel is stored in memory)
 - MKL_INTERFACE_LAYER=ILP64 python3 gnn/infinite_width/run.py --datasets "cora citeseer" --models "nn"
 - MKL_INTERFACE_LAYER=ILP64 python3 gnn/infinite_width/run.py --datasets "cora citeseer" --models "gnn"
 - MKL_INTERFACE_LAYER=ILP64 python3 gnn/infinite_width/run.py --datasets "cora citeseer" --models "snn"
 - MKL_DYNAMIC=FALSE MKL_INTERFACE_LAYER=ILP64 python3 gnn/infinite_width/run.py --datasets "cornell" --models "gat"
ef stands for effective resistance (how many percent of edges should be deleted) and is only used for gat + regression
### Reproducing Results
 - MKL_INTERFACE_LAYER=ILP64 python3 gnn/infinite_width/run.py --datasets "cora citeseer pubmed" --models "nn gnn sgnn"
 - MKL_INTERFACE_LAYER=ILP64 python3 gnn/infinite_width/run.py --datasets "cora" --models "gat"
 - MKL_INTERFACE_LAYER=ILP64 python3 gnn/infinite_width/run.py --datasets "citeseer" --models "gat"
 - MKL_INTERFACE_LAYER=ILP64 python3 gnn/infinite_width/run.py --datasets "pubmed" --models "gat" --ef 0.9
 - MKL_INTERFACE_LAYER=ILP64 python3 gnn/infinite_width/run.py --datasets "chameleon" --models "gat" --ef 0.9
 - MKL_INTERFACE_LAYER=ILP64 python3 gnn/infinite_width/run.py --datasets "squirrel" --models "gat" --ef 0.9
 - MKL_INTERFACE_LAYER=ILP64 python3 gnn/infinite_width/run.py --datasets "crocodile" --models "gat" --ef 0.9

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



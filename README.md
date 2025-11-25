Packages that are needed including `pytorch-geometric`, `pytorch`, and `pybind11`.
The code are tested under `cuda113` and `cuda116` environment. Please consider download these packages with the following commands:

```
# pytorch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# pytorch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

# pybind11 (used for c++ sampler)
pip install pybind11
```



## Run the code

Step 1: Compile C++ sampler (from https://github.com/amazon-science/tgl).
```
python setup.py build_ext --inplace
```

Step 2:  Preprocess data (from https://github.com/amazon-research/tgl)
```
python gen_graph.py --data REDDIT
Please replace `REDDIT` to other datasets, e.g., `WIKI`, `MOOC`, and`LASTFM`.
```
Step 3:
```
python make_split.py --data REDDIT  
```

Step 3: Run experiment
```
python train.py   --data WIKI   --num_neighbors 10   --use_onehot_node_feats    --rl_on   --use_grad   --grad_file DATA/WIKI/edge_list_grad.npy
```

You can use '--rl_on' to control the switch of the reinforcement learning selector and '--use_grad' to control whether to use a linear gradient
If you are running this dataset for the first, it need to take sometime pre-processing the input data. But it will only do it once.


import torch.nn as nn
import torch_geometric.nn as pyg_nn
from loguru import logger

def gin_module(in_channels, out_channels, eps=0, train_eps=False, layers=2, dropout=0.0, use_bn=False, aggr="add"):
    mlp = nn.Sequential(
        nn.Linear(in_channels, out_channels),
    )
    if use_bn:
      logger.debug(f"Using batch norm in GIN module")
      mlp.add_module(f"bn_input", nn.BatchNorm1d(out_channels))
    for _ in range(layers-1):
      mlp.add_module(f"relu_{_}", nn.ReLU())
      mlp.add_module(f"linear_{_}", nn.Linear(out_channels, out_channels))
      if use_bn:
        logger.debug(f"Using batch norm in GIN module")
        mlp.add_module(f"bn_{_}", nn.BatchNorm1d(out_channels))
    if dropout > 0:
      mlp.add_module(f"dropout", nn.Dropout(dropout))
    return pyg_nn.GINConv(mlp, eps, train_eps, aggr=aggr)

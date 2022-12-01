import torch
from torch import nn
from torch_geometric import nn as gnn
from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size

class CIConv(MessagePassing):
    def __init__(self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "add",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        self.aggr_module.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)
            
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = torch.mul(out,self.lin_r(x_r))
            #out = out+self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
    
        '''super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.out_channels = out_channels
        self.in_channels = (in_channels, in_channels)
        bias = True,
        self.lin_l = Linear(self.in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(self.in_channels[1], out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        #self.lin_r.reset_parameters()
    
    def forward(self, x, edge_index):
        """"""
        # propagate_type:  
        out = self.propagate(edge_index, x=x)
        #computes message x_j
        #x = self.lin_l(x)
        #out = self.lin_r(out)
        out = torch.mul(x, out)
        out = self.lin_l(out)
        # out = out + x
        #out = F.normalize(out, p=2., dim=-1)
        return out


    def message(self, x_j):
        return x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')'''


    

class CIGraphNN(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(self,
               feature_size,
               hidden_sizes,
              global_sizes):
    """Create the MLP.
    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
    """
    super(CIGraphNN, self).__init__()
    hidden_conv,hidden_final = hidden_sizes[0], hidden_sizes[1]
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #2 GAT layer
    self.conv1 = CIConv(feature_size, hidden_conv[0]) 
    self.conv1bn = gnn.BatchNorm(hidden_conv[0])
    self.conv2 = CIConv(hidden_conv[0],hidden_conv[1]) 
    self.conv2bn = gnn.BatchNorm(hidden_conv[1])
    self.conv3 = CIConv(hidden_conv[1],hidden_conv[2]) 
    self.conv3bn = gnn.BatchNorm(hidden_conv[2])

    #Join to Layer
    self.linear2 = nn.Linear(hidden_final[0], hidden_final[1])
    self.batchnorm1 = nn.BatchNorm1d(hidden_final[1])
    self.linear3 = nn.Linear(hidden_final[1], 1)
  
  def forward(self, node_feature, edge_index, global_x):
    x, edge_index= node_feature.to(self.device), edge_index.to(self.device)
    #GAT Layer for Graphs
    x = F.relu(self.conv1(x, edge_index))
    x = self.conv1bn(x)
    x = F.relu(self.conv2(x, edge_index))
    x = self.conv2bn(x)
    x = F.relu(self.conv3(x, edge_index))
    x = self.conv3bn(x)
    #x = F.relu(self.linear2(x))
    #x = self.batchnorm1(x)
    #x = F.relu(self.linear3(x))
    x = torch.softmax(x,dim=0)
    return x
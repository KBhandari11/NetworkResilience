import torch
from torch import nn
from torch_geometric import nn as gnn
from torch.nn import functional as F
from torch.nn import Linear
from torch_sparse import matmul
from torch_geometric.nn import MessagePassing

class CIConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.out_channels = out_channels
        self.in_channels = (in_channels, in_channels)
        bias = True,
        self.lin_l = Linear(self.in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(self.in_channels[1], out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
    
    def forward(self, x, edge_index):
        """"""
        # propagate_type:   
        out = self.propagate(edge_index, x=x)
        out = self.lin_l(out)

        x_r = x[1]
        out = torch.mul(out,self.lin_r(x_r))
        #out = out + self.lin_r(x_r)

        out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t,x):
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


    

class CIGraphNN(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(self,
               feature_size,
               hidden_sizes,global_sizes):
    """Create the MLP.
    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
    """
    super(CIGraphNN, self).__init__()
    hidden_conv,hidden_final = hidden_sizes[0], hidden_sizes[1]
    
    #2 GAT layer
    self.conv1 = CIConv(feature_size, hidden_conv[0]) 
    self.conv1bn = gnn.BatchNorm(hidden_conv[0])
    self.conv2 = CIConv(hidden_conv[0],hidden_conv[1]) 
    self.conv2bn = gnn.BatchNorm(hidden_conv[1])
    

    #Join to Layer
    self.linear2 = nn.Linear(hidden_final[0], hidden_final[1])
    self.batchnorm1 = nn.BatchNorm1d(hidden_final[1])
    self.linear3 = nn.Linear(hidden_final[1], 1)
  
  def forward(self, node_feature, edge_index, global_x):
    x, edge_index = node_feature, edge_index
    
    #GAT Layer for Graphs
    x = F.relu(self.conv1(x, edge_index))
    x = self.conv1bn(x)
    x = F.relu(self.conv2(x, edge_index))
    x = self.conv2bn(x)
    x = F.relu(self.linear2(x))
    x = self.batchnorm1(x)
    x = F.relu(self.linear3(x))
    x = torch.softmax(x,dim=0)
    return x
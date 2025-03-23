"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.gated_gcn_net import GatedGCNNet
from nets.gcn_net import GCNNet
from nets.gat_net import GATNet
from nets.graphsage_net import GraphSageNet
from nets.gin_net import GINNet
from nets.mo_net import MoNet as MoNet_
from nets.mlp_net import MLPNet
from nets.ring_gnn_net import RingGNNNet
from nets.three_wl_gnn_net import ThreeWLGNNNet
from nets.prgnn_net import PRGNNNet
from nets.li_net import LINet
from nets.brain_net_cnn import BrainNetCNN
from pooling.sagpool import SAGPoolReadout
from pooling.hgpslpool import HGPSLPoolReadout
from nets.transformer import GraphTransformer
from nets.gxn_net import GraphClassifier

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def RingGNN(net_params):
    return RingGNNNet(net_params)

def ThreeWLGNN(net_params):
    return ThreeWLGNNNet(net_params)

def PRGNN(net_params):
    return PRGNNNet(net_params)

def LI(net_params):
    return LINet(net_params)

def BrainCNN(net_params):
    return BrainNetCNN(net_params)

def SAGPool(net_params):
    return SAGPoolReadout(net_params)

def HGPSLPool(net_params):
    return HGPSLPoolReadout(net_params)

def Transformer(net_params):
    return GraphTransformer(net_params)

def GXN(net_params):
    return GraphClassifier(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet_,
        'MLP': MLP,
        'RingGNN': RingGNN,
        '3WLGNN': ThreeWLGNN,
        'PRGNN': PRGNN,
        'LINet': LI,
        'BrainNetCNN': BrainCNN,
        'SAGPool': SAGPool,
        'HGPSLPool': HGPSLPool,
        'Transformer': Transformer,
        'GXN': GXN
    }
    model = models[MODEL_NAME](net_params)
    model.name = MODEL_NAME
        
    return model

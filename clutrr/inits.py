import math
import torch
from torch import Tensor
from torch_geometric.data import Data

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(1.0 / (tensor.size(-2) + tensor.size(-1)))
        # print(stdv)
        tensor.data.uniform_(-stdv, stdv)
        # tensor.data.uniform_(-1, 1)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (Tensor, Tensor) -> Tensor
    pass


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout='coo')

def modifySubset(subset, pred_list, n_rel):
    new_subset = MySubset(subset, pred_list, n_rel)
    return new_subset

import torch
import networkx as nx
import torch_geometric.data

def to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.DiGraph` if :attr:`to_undirected` is set to :obj:`True`, or
    an undirected :obj:`networkx.Graph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.MultiDiGraph()

    G.add_nodes_from(range(data.num_nodes))

    values = {}
    for key, item in data:
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G

def find_edge_chains(s, edge_path):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    (edge_index,edge_type) = s
    r_path_dict = {}
    n_path_dict = {}

    i=0
    for p_num, ep in enumerate(edge_path):

        rel_path = []
        node_path = []
        for j, e in enumerate(ep):
            e = torch.LongTensor(e).to(device)[:2]
            node_path.append(list(e))

            match = torch.transpose(edge_index,0,1) == e
            idx = (match[:,0] & match[:,1]).nonzero(as_tuple = False)

            rel_path.append(edge_type[idx].view(-1,))

        _v = [str(x) for x in r_path_dict.values()]
        try:
            if str(torch.LongTensor(rel_path).to(device)) not in _v:
                r_path_dict[i] = torch.LongTensor(rel_path).to(device)
                n_path_dict[i] = torch.transpose(torch.LongTensor(node_path).to(device),0,1)
                i += 1
        except TypeError:

            all_r_path_comb = torch.cartesian_prod(*rel_path).to(device)

            for r_path in all_r_path_comb:
                if str(r_path) not in _v:
                    r_path_dict[i] = r_path.to(device)
                    n_path_dict[i] = torch.transpose(torch.LongTensor(node_path).to(device),0,1)
                    i += 1

    return n_path_dict, r_path_dict

def get_rel_path(s, Q):

    (edge_index,edge_type) = s
    x = torch.LongTensor(range(torch.max(edge_index)+1))
    G = to_networkx(Data(x=x, edge_index=edge_index))

    cut = 2
    edge_path = nx.all_simple_edge_paths(G, source=int(Q[0]), target=int(Q[1]), cutoff=cut)

    while next(edge_path, -1) == -1:
        cut += 1
        edge_path = nx.all_simple_edge_paths(G, source=int(Q[0]), target=int(Q[1]), cutoff=cut)

    edge_path = nx.all_simple_edge_paths(G, source=int(Q[0]), target=int(Q[1]), cutoff=cut)
    n_path_dict, r_path_dict = find_edge_chains(s, edge_path)

    return n_path_dict, r_path_dict

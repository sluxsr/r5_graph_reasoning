import copy
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData

from graphlog.types import DataLoaderType, GraphType
from graphlog import GraphLog
import itertools

from inits import *

class GetDataset(Dataset):
    def __init__(self, data_dir, data_world, mode):
        # gl = GraphLog()
        self.data_world = data_world
        self.data_loc = os.path.join(data_dir, data_world)
        self.data_dir = data_dir
        self.mode = mode

        self.data_rows: List[GraphRow] = []
        self.graphs: List[GeometricData] = []
        self.queries: List[Tuple[int,int]] = []
        self.labels: List[int] = []
        self.label_set: Set[int] = set()
        self.path_len: List[int] = []
        self.rules = []
        self.resolution_paths = []
        self.descriptors = []

        self.preds: List[int] = []

        self.get_label2id()
        self.load_graphs()
        self.load_dataset()

    def __getitem__(self, index):
        if self.mode != "train" and self.mode != "test" and self.mode != "valid":
            raise AssertionError("mode should be either train or test")
        return self.data_rows[index]

    def __len__(self) -> int:
        return len(self.data_rows)

    def get_label2id(self) -> Dict[str, int]:
        label2id_loc = os.path.join(
            self.data_dir, "label2id.json"
        )
        label2id = json.load(open(label2id_loc))
        assert isinstance(label2id, dict)
        self.label2id = label2id

    def load_graphs(self):

        train_raw_data = []
        valid_raw_data = []
        test_raw_data = []
        train_loc = os.path.join(self.data_loc,"train.jsonl")
        valid_loc = os.path.join(self.data_loc,"valid.jsonl")
        test_loc = os.path.join(self.data_loc,"test.jsonl")

        # load train
        with open(train_loc, "r") as fp:
            for line in fp:
                train_raw_data.append(line)
                    
        train_data = set(train_raw_data)
        # train_data = train_raw_data

        train_graphs = []
        for line in train_data:
            train_graphs.append(json.loads(line))

        # load valid
        with open(valid_loc, "r") as fp:
            for line in fp:
                valid_raw_data.append(line)
        valid_data = set(valid_raw_data) - set(train_data)
        # valid_data = valid_raw_data

        valid_graphs = []
        for line in valid_data:
            valid_graphs.append(json.loads(line))
        

        # load test
        with open(test_loc, "r") as fp:
            for line in fp:
                test_raw_data.append(line)
        # test_data = set(test_raw_data) - train_data - valid_data
        test_data = test_raw_data

        test_graphs = []
        for line in test_data:
            test_graphs.append(json.loads(line))
        
        self.json_graphs={}
        self.json_graphs["train"] = train_graphs
        self.json_graphs["valid"] = valid_graphs
        self.json_graphs["test"] = test_graphs

    def load_dataset(self):

        for gi, gs in enumerate(self.json_graphs[self.mode]):

            # graphs
            node2id: Dict[str, int] = {}
            edges = []
            edge_attr = []
            
            edge_set = gs["edges"]
            edge_set.sort() 
            edge_set = list(edge_set for edge_set,_ in itertools.groupby(edge_set))
            for (src, dst, rel) in edge_set:
                if src not in node2id:
                    node2id[src] = len(node2id)
                if dst not in node2id:
                    node2id[dst] = len(node2id)
                edges.append([node2id[src], node2id[dst]])
                target = self.label2id[rel]
                edge_attr.append(target)
            
            (src, dst, rel) = gs["query"]
            self.queries.append((node2id[src], node2id[dst]))
            target = self.label2id[rel]
            self.labels.append(target)
            self.label_set.add(target)

            _rpath = []
            for pth in gs["resolution_path"]:
                _rpath.append(node2id[pth])
            
            self.resolution_paths.append(_rpath)

            # # rules
            # _rules = []
            # for r in gs["rules_used"]:
            #     rels = r.split(",")
            #     _rule =[]
            #     for rel in rels:
            #         _rule.append(self.label2id[rel])
            #     _rules.append(_rule)
            # self.rules.append(_rules)

            # # descriptors
            # _descrips = []
            # desc = gs["descriptor"].split(",")
            # for r in desc:
            #     _descrips.append(self.label2id[r])
            # self.descriptors.append(_descrips)


            x = torch.arange(len(node2id)).unsqueeze(1)

            edge_index = list(zip(*edges))
            edge_index = torch.LongTensor(edge_index)  # type: ignore

            # 2 x num_edges
            assert edge_index.dim() == 2  # type: ignore
            geo_data = GeometricData(
                x=x,
                edge_index=edge_index,
                edge_attr=torch.tensor(edge_attr),
                y=torch.tensor([target]),
            )
            self.graphs.append(geo_data)

        graphRows: List[GraphRow] = []
        for i in range(len(self.graphs)):
            graphRows.append(
                GraphRow(
                    self.graphs[i],
                    self.queries[i],
                    self.labels[i],
                    # rule = self.rules[i],
                    # descriptor = self.descriptors[i],
                    resolution_path = self.resolution_paths[i],
                    rule = self.rules,
                    descriptor = self.descriptors,
                    pred = self.preds,
                )
            )
        self.data_rows = graphRows

class GraphRow:
    """Single row of information
    """
    def __init__(
        self,
        graph: GeometricData,
        query: np.ndarray,
        label: np.int64,
        resolution_path, 
        rule,
        descriptor,
        pred,
        edge_graph: Optional[GeometricData] = None,
    ):
        self.graph = graph
        self.edge_graph = edge_graph
        self.query = query
        self.label = label
        self.resolution_path = resolution_path
        self.rule = rule
        self.descriptor = descriptor
        self.pred = pred

class GraphBatch:
    """
    Batching class
    """
    def __init__(
        self,
        graphs: List[GeometricData],
        queries: Tensor,
        targets: Tensor,
        resolution_paths, 
        rules,
        descriptors,
        preds,
        device = 'cuda:0',
    ):
        self.num_nodes = [g.num_nodes for g in graphs]
        self.graphs = GeometricBatch.from_data_list(graphs)
        self.queries = torch.LongTensor(queries)  # type: ignore
        self.targets = targets.long()
        self.resolution_paths = resolution_paths
        self.rules = rules
        self.descriptors = descriptors
        self.preds = preds
        self.device = device

    def to(self, device: str) -> Any:
        self.device = device
        self.graphs = self.graphs.to(device)
        self.queries = self.queries.to(device)  # type: ignore
        self.targets = self.targets.to(device)
        return self

def pyg_collate(data: List[GraphRow]) -> GraphBatch:

    graphs = []
    queries = torch.zeros(len(data), 2).long()
    labels = torch.zeros(len(data)).long()
    rules = []
    descriptors = []
    resolution_paths = []
    preds = None

    for id, d in enumerate(data):

        graphs.append(d.graph)
        
        rules.append(d.rule)
        # print(d.rule)
        descriptors.append(d.descriptor)
        resolution_paths.append(d.resolution_path)

        queries[id][0] = torch.LongTensor([d.query[0]])  # type: ignore
        queries[id][1] = torch.LongTensor([d.query[1]])  # type: ignore
        labels[id] = torch.LongTensor([d.label])  # type: ignore

        preds=list(set(d.pred))

    return GraphBatch(
        graphs= graphs, queries=queries, targets=labels,
        resolution_paths = resolution_paths, 
        rules = rules, descriptors = descriptors,
        preds = preds,
    )
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
Example template for defining a system.
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from graphlog import GraphLog
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
from torch_geometric.nn import RGCNConv


class SupervisedRGCN(LightningModule):
    """
    Sample model to show how to define a template.
    """

    def __init__(self, hparams):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the model.
        """
        # init superclass
        super().__init__()
        self.hparams = hparams

        self.batch_size = hparams.rgcn_batch_size

        # if you specify an example input, the summary will show input/output for each layer
        # self.example_input_array = torch.rand(5, 28 * 28)

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout the model.
        """
        self.rgcn_layers = []
        for l in range(self.hparams.num_layers):
            in_channels = self.hparams.relation_embedding_dim
            out_channels = self.hparams.relation_embedding_dim
            num_bases = self.hparams.relation_embedding_dim

            self.rgcn_layers.append(
                RGCNConv(
                    in_channels,
                    out_channels,
                    self.hparams.num_classes,
                    num_bases,
                    root_weight=self.hparams.root_weight,
                    bias=self.hparams.bias,
                )
            )

        self.rgcn_layers = nn.ModuleList(self.rgcn_layers)
        self.classfier = []
        inp_dim = (
            self.hparams.relation_embedding_dim * 2
            + self.hparams.relation_embedding_dim
        )
        outp_dim = self.hparams.hidden_dim
        for l in range(self.hparams.classify_layers - 1):
            self.classfier.append(nn.Linear(inp_dim, outp_dim))
            self.classfier.append(nn.ReLU())
            inp_dim = outp_dim
        self.classfier.append(nn.Linear(inp_dim, self.hparams.num_classes))
        self.classfier = nn.Sequential(*self.classfier)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, batch):
        """
        We use random node embeddings for each forward call.
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        data = batch.graphs
        # initialize nodes randomly
        node_emb = torch.Tensor(
            size=(self.hparams.num_nodes, self.hparams.relation_embedding_dim)
        ).to(data.x.device)
        torch.nn.init.xavier_uniform_(node_emb, gain=1.414)
        x = F.embedding(data.x, node_emb)
        x = x.squeeze(1)

        # get edge attributes
        edge_types = (data.edge_attr).to(device)
        edge_index = data.edge_index.to(device)

        for nr in range(self.hparams.num_layers - 1):
            x = F.dropout(x, p=self.hparams.dropout, training=self.training)
            x = self.rgcn_layers[nr](x, edge_index, edge_types)
            x = F.relu(x)
        x = self.rgcn_layers[self.hparams.num_layers - 1](
            x, edge_index, edge_types
        )
        # restore x into B x num_node x dim
        chunks = torch.split(x, batch.num_nodes, dim=0)
        chunks = [p.unsqueeze(0) for p in chunks]
        # x = torch.cat(chunks, dim=0)
        # classify
        query_emb = []
        for i in range(len(chunks)):
            query = (
                batch.queries[i]
                .unsqueeze(0)
                .unsqueeze(2)
                .repeat(1, 1, chunks[i].size(2))
            )  # B x num_q x dim
            query_emb.append(torch.gather(chunks[i], 1, query))
        query_emb = torch.cat(query_emb, dim=0)
        query = query_emb.view(query_emb.size(0), -1)  # B x (num_q x dim)
        # pool the nodes
        # mean pooling
        node_avg = torch.cat(
            [torch.mean(chunks[i], 1) for i in range(len(chunks))], dim=0
        )  # B x dim
        # concat the query
        edges = torch.cat((node_avg, query), -1)  # B x (dim + dim x num_q)
        return self.classfier(edges)


    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Parameters you define here will be available to your model through `self.hparams`.
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        # parser.add_argument("--train_world", default="rule_0", type=str)
        parser.add_argument("--num_layers", default=4, type=int)
        parser.add_argument(
            "--num_classes", default=21, type=int, help="20 classes including UNK rel"
        )
        parser.add_argument("--relation_embedding_dim", default=200, type=int)
        parser.add_argument("--root_weight", default=False, action="store_true")
        parser.add_argument("--bias", default=False, action="store_true")
        parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--classify_layers", default=2, type=int)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument("--hidden_dim", default=50, type=int)
        parser.add_argument("--rgcn_learning_rate", default=0.001, type=float)
        parser.add_argument(
            "--num_nodes", default=10000, type=int, help="Set a max number of nodes"
        )

        # data
        parser.add_argument(
            "--data_root", default=os.path.join(root_dir, "mnist"), type=str
        )

        # training params (opt)
        parser.add_argument("--rgcn_epochs", default=20, type=int)
        parser.add_argument("--optimizer_name", default="adam", type=str)
        parser.add_argument("--rgcn_batch_size", default=1, type=int)
        return parser
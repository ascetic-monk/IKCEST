# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PGL Graph
"""
import sys
import os
import numpy as np
import pandas as pd

from pgl.graph import Graph


def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    """Load weight matrix function."""
    try:
        W = np.load(file_path)
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 100.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (
            np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W


class GraphFactory(object):
    """GraphFactory"""

    def __init__(self, args):
        self.args = args
        self.adj_matrix = weight_matrix(self.args.adj_mat_file, scaling=False)

        L = np.eye(self.adj_matrix.shape[0]) + self.adj_matrix
        D = np.sum(self.adj_matrix, axis=1)

        edges = []
        weights = []
        for i in range(self.adj_matrix.shape[0]):
            for j in range(self.adj_matrix.shape[1]):
                edges.append([i, j])
                # weights.append(L[i][j])
        weights = np.reshape(L, (-1, 1))
        self.edges = np.array(edges, dtype=np.int64)
        self.weights = np.array(weights, dtype=np.float32).reshape(-1, 1)

        self.norm = np.zeros_like(D, dtype=np.float32)
        self.norm[D > 0] = np.power(D[D > 0], -0.5)
        self.norm = self.norm.reshape(-1, 1)

    def build_graph(self, x_batch, idx=0, Kt=4, rande=False):
        """build graph"""
        B, T, n, _ = x_batch.shape
        dide = [Kt-1, 3*(Kt-1)]
        T = T - dide[idx]

        batch = B * T

        batch_edges = []
        for i in range(batch):
            batch_edges.append(self.edges + (i * n))
        batch_edges = np.vstack(batch_edges)

        num_nodes = B * T * n
        if rande:
            weights, norm = self.randgraph()
        else:
            weights = self.weights
            norm = self.norm
        node_feat = {'norm': np.tile(norm, [batch, 1])}
        edge_feat = {'weights': np.tile(weights, [batch, 1])}
        graph = Graph(
            num_nodes=num_nodes,
            edges=batch_edges,
            node_feat=node_feat,
            edge_feat=edge_feat)

        return graph

    def randgraph(self):
        adj0, adj1 = self.adj_matrix.shape
        randw = (1 + 0.4 * (np.random.rand(adj0, adj1) - 0.5))
        adj_matrix = self.adj_matrix * randw
        L = np.eye(adj_matrix.shape[0]) + adj_matrix
        D = np.sum(adj_matrix, axis=1)

        weights = np.reshape(L, (-1, 1)).astype(np.float32)

        norm = np.zeros_like(D, dtype=np.float32)
        norm[D > 0] = np.power(D[D > 0], -0.5)
        norm = norm.reshape(-1, 1)
        return weights, norm

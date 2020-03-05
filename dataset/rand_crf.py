import os
import glob
import torch
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.sparse import coo_matrix

from utils.topology import get_msg_graph
from torch.utils.data import Dataset
from utils.data_helper import *

__all__ = ['RandCRFData']


class RandCRFData(Dataset):

  def __init__(self, config, split='train'):
    assert split in ['train', 'val', 'test_I', 'test_II', 'test_III', 'test_IV', 'test_V', 'test_VI', 'test_VII',
                     'test_VIII'], "no such split"
    self.config = config
    self.split = split
    self.data_path = config.dataset.data_path
    self.data_files = sorted(glob.glob(os.path.join(self.data_path, split, '*.p')))
    self.num_graphs = len(self.data_files)
    self.npr = np.random.RandomState(seed=config.seed)
  def __getitem__(self, index):
    graph = pickle.load(open(self.data_files[index], 'rb'))
    if 'prob_gt' not in graph.keys():
      graph['prob_gt'] = np.stack([graph['prob_hmc'], 1-graph['prob_hmc']], axis=1)

    # idx1 = self.data_files[index].find('graph_')
    # idx2 = self.data_files[index][idx1+6:].find('_')
    # graph['topology'] = self.data_files[index][idx1+6:][:idx2]
    #
    # # Added by Kijung on 12/25/2019
    # if 'adj' not in graph.keys():
    #   from utils.topology import NetworkTopology, get_msg_graph
    #   topology = NetworkTopology(num_nodes=len(graph['b']), seed=self.config.seed)
    #   G, _ = topology.generate(topology=graph['topology'])
    #   graph['adj'] = topology.graph_to_adjacency_matrix(G)
    #   msg_node, msg_adj = get_msg_graph(G)
    #   graph['msg_node'] = msg_node
    #   graph['msg_adj'] = np.asarray(msg_adj)
    #   graph['J'] = graph['J'].todense()

    if self.config.model.name == 'TreeReWeightedMessagePassing':
      A = graph['adj']
      graph['prob_gt'] = torch.from_numpy(graph['prob_gt']).float()
      graph['adj'] = torch.from_numpy(graph['adj']).float()
      graph['J'] = torch.from_numpy(graph['J']).float()
      graph['b'] = torch.from_numpy(graph['b']).float()

      msg_node, msg_adj = [], []
      for ii in range(self.config.model.num_trees):
        W = self.npr.rand(A.shape[0], A.shape[0])
        W = np.multiply(W, A)
        G = nx.from_numpy_matrix(W).to_undirected()
        T = nx.minimum_spanning_tree(G)
        msg_node_tmp, msg_adj_tmp = get_msg_graph(T)
        msg_node += [msg_node_tmp]
        msg_adj += [msg_adj_tmp]

      graph['msg_node'] = torch.stack(
          [torch.from_numpy(np.array(xx)).long() for xx in msg_node], dim=0)
      graph['msg_adj'] = torch.stack(
          [torch.from_numpy(xx).float() for xx in msg_adj], dim=0)
    else:
      pass
      # graph['prob_gt'] = torch.from_numpy(graph['prob_gt']).float()
      # graph['adj'] = torch.from_numpy(graph['adj']).float()
      # graph['J'] = torch.from_numpy(graph['J']).float()
      # graph['b'] = torch.from_numpy(graph['b']).float()
      # graph['msg_node'] = torch.from_numpy(np.array(graph['msg_node'])).long()
      # graph['msg_adj'] = torch.from_numpy(graph['msg_adj']).float()

    return graph

  def __len__(self):
    return self.num_graphs

  def collate_fn(self, batch): # batch : list of dicts
    assert isinstance(batch, list)
    data = {}

    # if 'msg_node' not in batch[0].keys():
    #   data['prob_gt'] = torch.from_numpy(
    #     np.concatenate([bch['prob_gt'] for bch in batch], axis=0)).float()
    #
    #   data['b'] = torch.from_numpy(
    #     np.concatenate([bch['b'] for bch in batch], axis=0)).float()
    #
    #   n = data['b'].shape[0]
    #   data['J'] = coo_matrix(np.zeros([n, n]))
    #
    #   pad_size_l = np.array([bch['J'].shape[0] for bch in batch]).cumsum()
    #   pad_size_r = pad_size_l
    #   pad_size_r = pad_size_r[-1] - pad_size_r
    #   pad_size_l = np.concatenate(([0], pad_size_l[:-1]))
    #
    #   data['J'] = torch.from_numpy(
    #     np.stack(
    #     [
    #       np.pad(bch['J'].todense(), (pad_size_l[ii], pad_size_r[ii]), 'constant', constant_values=0.0) for ii, bch in enumerate(batch)
    #     ],
    #     axis=0).sum(axis=0)).float()
    #   G = nx.from_numpy_array(data['J'].numpy())
    #
    #   # row = []
    #   # col = []
    #   # val = []
    #   # for ii, bch in enumerate(batch):
    #   #   nv = bch['J'].shape[0]
    #   #   row.append(bch['J'].row + nv * ii)
    #   #   col.append(bch['J'].col + nv * ii)
    #   #   val.append(bch['J'].data)
    #   #
    #   # data['J'].row = np.concatenate(row)
    #   # data['J'].col = np.concatenate(col)
    #   # data['J'].data = np.concatenate(val)
    #   # G = nx.from_scipy_sparse_matrix(data['J'])
    #   #
    #   # values = torch.FloatTensor(data['J'].data)
    #   # idx = torch.LongTensor(np.vstack((data['J'].row, data['J'].col)))
    #   # data['J'] = torch.sparse.FloatTensor(idx, values, torch.Size(data['J'].shape))
    #
    #   msg_node, msg_adj = get_msg_graph(G)
    #   data['msg_node'] = torch.from_numpy(np.array(msg_node)).long()
    #   data['msg_adj'] = torch.from_numpy(np.array(msg_adj)).float()

    data['prob_gt'] = torch.from_numpy(
      np.concatenate([bch['prob_gt'] for bch in batch], axis=0)).float()
    data['J_msg'] = torch.from_numpy(
      np.concatenate([bch['J_msg'] for bch in batch], axis=0)).float()
    data['b'] = torch.from_numpy(
      np.concatenate([bch['b'] for bch in batch], axis=0)).float()

    idx_msg_edge = np.empty((0, 2))
    msg_node = np.empty((0, 2))
    num_msg_node = 0
    for bch in batch:
      idx_msg_edge = np.vstack((idx_msg_edge, msg_node.shape[0] + bch['idx_msg_edge']))
      msg_node = np.vstack((msg_node, num_msg_node + bch['msg_node']))
      num_msg_node = 1 + msg_node.max()
    data['msg_node'] = torch.from_numpy(msg_node).long()
    data['idx_msg_edge'] = torch.from_numpy(idx_msg_edge).long()

    # values = torch.FloatTensor(batch[0]['J'].data)
    # idx = torch.LongTensor(np.vstack((batch[0]['J'].row, batch[0]['J'].col)))
    # data['J'] = torch.sparse.FloatTensor(idx, values, torch.Size(batch[0]['J'].shape))

    # data['J'] = torch.from_numpy(batch[0]['J']).float()
    # data['msg_node'] = torch.from_numpy(batch[0]['msg_node']).long()
    # data['msg_adj'] = torch.from_numpy(batch[0]['msg_adj']).long()

    return data
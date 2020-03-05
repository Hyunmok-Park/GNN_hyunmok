import os
import pickle
import numpy as np
from utils.topology import NetworkTopology, get_msg_graph
from model.block_gibbs import BlockGibbs
from model.hmc import HMC
from model.min_cut import MinCut
from scipy.sparse import coo_matrix
import argparse

def mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('made directory {}'.format(dir_name))

def main(sample_id):
  """
    Random Potentials Class:
      A: J_ij = J_ji ~ N(0, 1), b_i ~ N(0, (1/4)^2)
      B: J_ij = J_ji ~ N(0, 0.5), b_i ~ N(0, (1/8)^2)

    Train Protocols:
      |V| = 9, 13 special structures, random potentials A
    Test Protocols:
      I:  |V| = 9, 13 special structures, random potentials A
      II: |V| = 100, 13 special structures, random potentials A
  """
  seed_test = int(str(3333)+str(sample_id))
  std_J_A = 3.0
  std_b_A = 0.25
  num_nodes_II = 100
  num_graphs_test = 10
  save_dir = '../data_temp/'

  gibb_burn_in = 100
  gibb_num_sample = 100
  block_size = 4

  hmc_burn_in = 100
  hmc_num_sample = 100
  trv_time = 49.5

  topology_list = [
      'star', 'binarytree', 'path', 'cycle', 'wheel', 'ladder',
      'circladder', 'grid', 'barbell', 'lollipop', 'bipartite',
      'tripartite', 'complete'
  ]
  npr = np.random.RandomState(seed=seed_test)

  #############################################################################
  # Generate Training Graphs
  #############################################################################
  # here seed only affects random graphs
  print('Generating training graphs!')
  topology = NetworkTopology(num_nodes=num_nodes_II, seed=seed_test)
  try:
    mkdir(os.path.join(save_dir, 'test_II'))
  except OSError:
    pass

  for tt in topology_list:
    print(tt)
    graph = {}
    G, W = topology.generate(topology=tt)
    J = npr.normal(0, std_J_A / np.sqrt(num_nodes_II), size=[num_nodes_II, num_nodes_II])
    J = (J + J.transpose()) / 2.0
    J = J * W
    b = npr.normal(0, std_b_A, size=[num_nodes_II, 1])

    # Block Gibbs Sampling
    model = BlockGibbs(G, J, b, block_method='gibbs', block_size=block_size, seed=seed_test)
    prob_gibbs = model.inference(burn_in=gibb_burn_in, num_sample=gibb_num_sample, sample_gap=50)
    # Hamiltonian Monte Carlo
    model = HMC(W, J, b, seed=seed_test)
    prob_hmc = model.inference(travel_time=trv_time, num_sample=hmc_num_sample, burn_in=hmc_burn_in)

    if (J >= 0).all():
        model = MinCut(G, J, b)
        map_gt = model.inference()
        graph['map_gt'] = map_gt  # shape N x 2

    graph['prob_gibbs'] = prob_gibbs  # shape N x 2
    graph['prob_hmc'] = prob_hmc
    graph['J'] = coo_matrix(J)  # shape N X N
    graph['b'] = b  # shape N x 1
    graph['seed_test'] = seed_test

    msg_node, msg_adj = get_msg_graph(G)
    msg_node, msg_adj = np.array(msg_node), np.array(msg_adj)
    idx_msg_edge = np.transpose(np.nonzero(msg_adj))
    J_msg = J[msg_node[:, 0], msg_node[:, 1]].reshape(-1, 1)

    graph['msg_node'] = msg_node
    graph['idx_msg_edge'] = idx_msg_edge
    graph['J_msg'] = J_msg

    file_name = os.path.join(save_dir, 'test_II', 'graph_{}_nn{}_{:07d}.p'.format(tt, num_nodes_II, sample_id))
    with open(file_name, 'wb') as f:
      pickle.dump(graph, f)
      del graph


# python3 -m dataset.gen_test_II
# Use different seed for train/val/test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_id', type=int, default=1, help='sample_id')
    args = parser.parse_args()

    main(args.sample_id)

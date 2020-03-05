import collections
import numpy as np
import networkx as nx


def get_msg_graph(G):
  L = nx.line_graph(G.to_directed())

  # remove redundant edges
  redundant_edges = []
  for edge in L.edges():
    if set(edge[0]) == set(edge[1]):
      redundant_edges.append(edge)

  for edge in redundant_edges:
    L.remove_edge(edge[0], edge[1])

  node_list = sorted(L.nodes)

  adj = nx.adjacency_matrix(L, nodelist=node_list)
  return node_list, adj.todense()


class NetworkTopology():

  def __init__(self, num_nodes, seed=1234):
    self.num_nodes = num_nodes
    self.seed = seed

  def generate(self, topology, p=None):
    if topology == 'star':
      G, W = self.star()
    elif topology == 'binarytree':
      G, W = self.binarytree()
    elif topology == 'path':
      G, W = self.path()
    elif topology == 'cycle':
      G, W = self.cycle()
    elif topology == 'wheel':
      G, W = self.wheel()
    elif topology == 'ladder':
      G, W = self.ladder()
    elif topology == 'circladder':
      G, W = self.circladder()
    elif topology == 'grid':
      G, W = self.grid()
    elif topology == 'barbell':
      G, W = self.barbell()
    elif topology == 'lollipop':
      G, W = self.lollipop()
    elif topology == 'bipartite':
      G, W = self.bipartite()
    elif topology == 'tripartite':
      G, W = self.tripartite()
    elif topology == 'complete':
      G, W = self.complete()
    elif topology == 'random':
      assert p is not None
      G, W = self.random_connected_np(p, seed=self.seed)
      self.seed += 1
    elif topology == 'samedegree':
      G, W = self.random_same_degree_with()
    elif topology == 'uniqdegree':
      G, W = self.random_unique_degree_of()
    else:
      raise Exception("network topology not supported: {}".format(topology))
    return G, W

  def graph_to_adjacency_matrix(self, G):
    n = G.number_of_nodes()
    W = np.zeros([n, n])
    for i in range(n):
      W[i, [j for j in G.neighbors(i)]] = 1
    return W

  def degree(self, G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    return degree_sequence, deg, cnt

  def unique_deg_preserved_seq(self, degree, count):
    num_node = sum(count)
    num_unique_deg = len(count)
    legitimate = False
    for i in range(100):
      cnt_new = np.random.randint(1, num_node - num_unique_deg + 1, size=num_unique_deg)
      if sum(cnt_new) == num_node and any(cnt_new != np.array(count)):
        legitimate = True
        break

    seq = []
    if legitimate:
      for i, c in enumerate(cnt_new):
        seq += [degree[i]] * c

    return seq

  # ----------- ACYCLIC GRAPH --------------
  def star(self):
    G = nx.star_graph(self.num_nodes - 1)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def binarytree(self):
    r = 2
    h = int(np.log2(self.num_nodes))
    G = nx.balanced_tree(r, h)
    for i in range(self.num_nodes, r**(1 + h) - 1):
      G.remove_node(i)

    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def path(self):
    G = nx.path_graph(self.num_nodes)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  # ----------- CYCLIC GRAPH --------------
  def cycle(self):
    G = nx.cycle_graph(self.num_nodes)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def wheel(self):
    G = nx.wheel_graph(self.num_nodes)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def ladder(self):
    n = int(self.num_nodes / 2)
    G = nx.ladder_graph(n)
    if self.num_nodes > 2 * n:
      G.add_edge(2 * n - 1, 2 * n)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def circladder(self):
    n_cycle = int(self.num_nodes / 2)
    G = nx.circular_ladder_graph(n_cycle)
    if self.num_nodes > 2 * n_cycle:
      G.add_edge(2 * n_cycle - 1, 2 * n_cycle)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def grid(self):
    sqrtN = int(np.sqrt(self.num_nodes))
    assert self.num_nodes == sqrtN**2
    G = nx.grid_2d_graph(sqrtN, sqrtN)
    W = np.zeros([self.num_nodes, self.num_nodes])
    nodes = [n for n in G.nodes]
    for i in range(self.num_nodes):
      for j in G.neighbors(nodes[i]):
        W[i, sqrtN * j[0] + j[1]] = 1

    node_map = {gg: ii for ii, gg in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, node_map)
    return G, W

  def barbell(self):
    m1 = self.num_nodes // 2
    m2 = self.num_nodes % 2
    G = nx.barbell_graph(m1, m2)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def lollipop(self):
    m = int(np.ceil(self.num_nodes / 2)) + 1
    n = self.num_nodes - m
    G = nx.lollipop_graph(m, n)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def bipartite(self):
    n1 = self.num_nodes // 2
    n2 = self.num_nodes - n1
    G = nx.complete_bipartite_graph(n1, n2)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def tripartite(self):
    n1 = self.num_nodes // 3
    n2 = self.num_nodes // 3
    n3 = self.num_nodes - n1 - n2
    G = nx.complete_multipartite_graph(n1, n2, n3)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def complete(self):
    G = nx.complete_graph(self.num_nodes)
    W = self.graph_to_adjacency_matrix(G)
    return G, W

  # ----------- RANDOM GRAPH --------------
  def random_connected_np(self, p, seed=1234):
    G = nx.fast_gnp_random_graph(self.num_nodes, p, seed=seed)
    cnt = 2
    while not nx.is_connected(G):
      G = nx.fast_gnp_random_graph(self.num_nodes, p, seed=seed * cnt)
      cnt += 1

    W = self.graph_to_adjacency_matrix(G)
    return G, W

  def random_same_degree_with(self, G, num_graphs):
    W = self.graph_to_adjacency_matrix(G)
    seq_deg, _, _ = self.degree(G)
    G_list = [G]
    W_list = [W]

    cnt = 0
    for i in range(1, 1000):
      legitimate = True
      try:
        G_new = nx.random_degree_sequence_graph(seq_deg)
        for j in range(len(G_list)):
          if nx.is_isomorphic(G_list[j], G_new) or not nx.is_connected(G_new):
            legitimate = False
            break
        if legitimate:
          G_list.append(G_new)
          W_list.append(self.graph_to_adjacency_matrix(G_new))
          cnt += 1
      except:
        pass

      if cnt == num_graphs:
        break

    return G_list[1:], W_list[1:]

  def random_unique_degree_of(self, G, num_graphs):
    W = self.graph_to_adjacency_matrix(G)
    _, deg, count = self.degree(G)
    G_list = [G]
    W_list = [W]

    cnt = 0
    for i in range(1, 1000):
      legitimate = True
      try:
        seq = self.unique_deg_preserved_seq(deg, count)
        G_new = nx.random_degree_sequence_graph(seq)
        for j in range(len(G_list)):
          if nx.is_isomorphic(G_list[j], G_new) or not nx.is_connected(G_new):
            legitimate = False
            break
        if legitimate:
          G_list.append(G_new)
          W_list.append(self.graph_to_adjacency_matrix(G_new))
          cnt += 1
      except:
        pass

      if cnt == num_graphs:
        break

    return G_list[1:], W_list[1:]

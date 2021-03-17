import networkx as nx
from node2vec import Node2Vec

import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv, GATConv
import pickle
import numpy as np

base_dir = "/home/zhiling/py3_workspace/freesound-audio-tagging-2019/input"
# graph_dir = '/home/zhiling/py3_workspace/freesound-audio-tagging-2019/input/audioset_aser_all_pre_conj_0.pkl'
for graph_type in ["audioset_all_direct", "audioset_aser_all_pre_conj_0", "aser_all_pre_conj_0"]:
    print(graph_type)
    graph_dir = f'{base_dir}/{graph_type}.pkl'
    with open(graph_dir, "rb") as f:
        graph, class_indices = pickle.load(f)
    graph = graph.remove_self_loop()
    graph = graph.to_networkx()
    node2vec = Node2Vec(graph, dimensions=512, workers=4, walk_length=30, num_walks=10)
    model = node2vec.fit(window=5, iter=30)

    # Look for most similar nodes
    print(model.wv.most_similar('2'))  # Output node names are always strings

    # Save embeddings for later use
    class_indices = [str(x) for x in class_indices]
    selected_embs = model.wv[class_indices]
    np.save(f"{base_dir}/{graph_type}_n2v.npy", selected_embs)
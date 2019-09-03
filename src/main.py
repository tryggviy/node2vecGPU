'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from word2vec_gpu import word2vec_basic
import sys
import os

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--gputrain', dest='gputrain', action='store_true',
	                    help='Boolean specifying how to train. Default is training with GPU.')
	parser.set_defaults(gputrain=True)

	parser.add_argument('--input', nargs='?', default='../graph/listed_weight.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='../emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=15,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=1000,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--start_learning_rate', type=int, default=0.025,
						help='the start learning rate')

	parser.add_argument('--end_learning_rate', type=int, default=0.0001,
						help='the end learning rate')

	parser.add_argument('--decay_power', type=int, default=0.25,
						help='decaying power of learning rate')

	parser.add_argument('--iter', default=10000, type=int,
                      	help='Number of epochs in SGD')

	parser.add_argument('--batch_size', type=int, default=100,
						help='Number of sample in one batch.')

	parser.add_argument('--num_sampled', type=int, default=64,
						help='Number of parallel workers. Default is 8.')

	parser.add_argument('--learning_rate', type=int, default=0.1,
						help='Learning rate for GradientDescentOptimizer')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')

	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=True)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=True)
	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)

def learn_embeddings_tensorflow(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
	para_dict = {}
	para_dict['log_dir'] = os.path.join(current_path, 'log')
	para_dict['batch_size'] = args.batch_size
	para_dict['embedding_size'] = args.dimensions
	para_dict['skip_window'] = args.window_size
	para_dict['num_sampled'] = args.num_sampled
	para_dict['num_steps'] = args.iter + 1
	para_dict['emb_path'] = args.output + '.emb'
	para_dict['start_learning_rate'] = args.start_learning_rate
	para_dict['end_learning_rate'] = args.end_learning_rate
	para_dict['decay_power'] = args.decay_power

	# These parameters are default settings.
	# They will not effect the model performance.
	para_dict['valid_size'] = 16
	para_dict['valid_window'] = 100
	para_dict['printsim'] = False
	para_dict['plot_pct'] = 0.3
	para_dict['plot_path'] = '../'

	# walks converted to str type
	walksinstr = [list(map(str, walk)) for walk in walks]

	print('embeding started...')
	word2vec_basic(para_dict, walksinstr)
	print('embeding finished...')

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph()
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)

	if args.gputrain:
		learn_embeddings_tensorflow(walks)
	else:
		learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)

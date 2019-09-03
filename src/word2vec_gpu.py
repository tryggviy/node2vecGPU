# This is a node2vec implementation with word2vec of tensorflow
# This normally speed up embedding process after generating the random walks
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import hashlib
import math
import os
import random
import sys
import zipfile
import pickle
import pandas as pd
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

data_index = 0


def word2vec_basic(para_dict, walksinstr):
	"""Example of building, training and visualizing a word2vec model."""
	# Create the directory for TensorBoard variables if there is not.
	if not os.path.exists(para_dict['log_dir']):
		os.makedirs(para_dict['log_dir'])
	print('number of walks', len(walksinstr))

	# Step 1: Build the dictionary with all nodes in the graph
	# walks are falattend to nodes
	flatwalks = ([x for l in walksinstr for x in l])

	# Total number of nodes in four graphs
	random_walks_size = len(set(flatwalks))

	def build_dataset(words, n_words):
		"""Process raw inputs into a dataset."""
		count = []
		count.extend(collections.Counter(words).most_common(n_words))
		dictionary = dict()
		for word, _ in count:
			dictionary[word] = len(dictionary)
		data = list()
		for word in words:
			index = dictionary[word]
			data.append(index)
		reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		return data, count, dictionary, reversed_dictionary

	# Filling 4 global variables:
	# data - list of codes (integers from 0 to random_walks_size-1).
	#   This is the original text but words are replaced by their codes
	# count - map of words(strings) to count of occurrences
	# dictionary - map of words(strings) to their codes(integers)
	# reverse_dictionary - map of codes(integers) to words(strings)
	data, count, recover_dictionary, reverse_dictionary = build_dataset(flatwalks, random_walks_size)

	print('Most common words (+UNK)', count[:5])
	print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

	# Step 3: Function to generate a training batch for the skip-gram model.

	# Generate all possible pairs of samples for skip-gram model
	def generate_graph_context_all_pairs(path, window_size):
		print('generating graph context pairs...')
		'''generating graph context pairs here. It may take a while'''
		all_pairs = []
		for k in range(len(path)):
			for i in range(len(path[k])):
				dynmc_window = random.randint(1, window_size)
				for j in range(i - dynmc_window, i + dynmc_window + 1):
					if i == j or j < 0 or j >= len(path[k]):
						continue
					else:
						all_pairs.append([path[k][i], path[k][j]])
			if k in range(0, len(path), int(len(path)*0.1)):
				print('generated '+str(int(k/len(path)*100))+'% pairs...')
		print('graph context pairs generated...')
		return np.array(all_pairs, dtype=np.int32)

	# Generate a batch of samples each time
	def generate_batch(all_pairs, batch_size):
		while True:
			start_idx = np.random.randint(0, len(all_pairs) - batch_size)
			batch_idx = np.array(range(start_idx, start_idx + batch_size))
			batch_idx = np.random.permutation(batch_idx)
			batch = np.zeros(batch_size, dtype=np.int32)
			labels = np.zeros((batch_size, 1), dtype=np.int32)
			batch[:] = all_pairs[batch_idx, 0]
			labels[:, 0] = all_pairs[batch_idx, 1]
			yield batch, labels

	# Step 4: Build and train a skip-gram model.

	batch_size = para_dict['batch_size']
	embedding_size = para_dict['embedding_size']  # Dimension of the embedding vector.
	skip_window = para_dict['skip_window']  # How many words to consider left and right.
	num_sampled = para_dict['num_sampled']  # Number of negative examples to sample.

	# We pick a random validation set to sample nearest neighbors. Here we limit
	# the validation samples to the words that have a low numeric ID, which by
	# construction are also the most frequent. These 3 variables are used only for
	# displaying model accuracy, they don't affect calculation.
	valid_size = para_dict['valid_size']  # Random set of words to evaluate similarity on.
	valid_window = para_dict['valid_window']  # Only pick dev samples in the head of the distribution.
	valid_examples = np.random.choice(valid_window, valid_size, replace=False)

	random_walksint = ([[recover_dictionary[i] for i in x] for x in walksinstr])
	del walksinstr  # Hint to reduce memory.
	allsamples = generate_graph_context_all_pairs(path=random_walksint, window_size=skip_window)

	batch, labels = next(generate_batch(all_pairs=allsamples, batch_size=8))
	for i in range(8):
		print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

	graph = tf.Graph()

	with graph.as_default():

		# Input data.
		with tf.name_scope('inputs'):
			train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
			train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
			valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

		# Ops and variables pinned to the CPU because of missing GPU implementation
		with tf.device('/cpu:0'):
			# Look up embeddings for inputs.
			with tf.name_scope('embeddings'):
				embeddings = tf.Variable(tf.random_uniform([random_walks_size, embedding_size], -1.0, 1.0))
				embed = tf.nn.embedding_lookup(embeddings, train_inputs)

			# Construct the variables for the NCE loss
			with tf.name_scope('weights'):
				nce_weights = tf.Variable(
					tf.truncated_normal([random_walks_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
			with tf.name_scope('biases'):
				nce_biases = tf.Variable(tf.zeros([random_walks_size]))

		# Compute the average NCE loss for the batch.
		# tf.nce_loss automatically draws a new sample of the negative labels each
		# time we evaluate the loss.
		# Explanation of the meaning of NCE loss and why choosing NCE over tf.nn.sampled_softmax_loss:
		#   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
		#   http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf
		with tf.name_scope('loss'):
			loss = tf.reduce_mean(
					tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
								   num_sampled=num_sampled, num_classes=random_walks_size))

		# Add the loss value as a scalar to summary.
		tf.summary.scalar('loss', loss)

		global_step = tf.Variable(0, trainable=False)
		starter_learning_rate = para_dict['start_learning_rate']
		end_learning_rate = para_dict['end_learning_rate']
		decay_steps = para_dict['num_steps']
		learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step, decay_steps, end_learning_rate,
												  power=para_dict['decay_power'])

		# Construct the SGD optimizer using a learning rate of 1.0.
		with tf.name_scope('optimizer'):
			optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

		# Compute the cosine similarity between minibatch examples and all
		# embeddings.
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
		similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

		# Merge all summaries.
		merged = tf.summary.merge_all()

		# Add variable initializer.
		init = tf.global_variables_initializer()

		# Create a saver.
		saver = tf.train.Saver()

	# Step 5: Begin training.
	num_steps = para_dict['num_steps']
	printsim = para_dict['printsim']

	# with tf.compat.v1.Session(graph=graph) as session: ---modified here
	with tf.Session(graph=graph) as session:
		# Open a writer to write summaries.
		writer = tf.summary.FileWriter(para_dict['log_dir'], session.graph)

		# We must initialize all variables before we use them.
		init.run()
		print('Initialized')

		average_loss = 0
		for step in xrange(num_steps):
			batch_inputs, batch_labels = next(generate_batch(all_pairs=allsamples, batch_size=batch_size))
			feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

			# Define metadata variable.
			run_metadata = tf.RunMetadata()

			# We perform one update step by evaluating the optimizer op (including it
			# in the list of returned values for session.run()
			# Also, evaluate the merged op to get all summaries from the returned
			# "summary" variable. Feed metadata variable to session for visualizing
			# the graph in TensorBoard.
			_, summary, loss_val = session.run([optimizer, merged, loss], feed_dict=feed_dict,
											   run_metadata=run_metadata)
			average_loss += loss_val
			# Add returned summaries to writer in each step.
			writer.add_summary(summary, step)
			# Add metadata to visualize the graph for the last run.
			if step == (num_steps - 1):
				writer.add_run_metadata(run_metadata, 'step%d' % step)

			if step % 2000 == 0:
				if step > 0:
					average_loss /= 2000
				# The average loss is an estimate of the loss over the last 2000
				# batches.
				print('Average loss at step ', step, ': ', average_loss)
				average_loss = 0

			# Note that this is expensive (~20% slowdown if computed every 500 steps)
			if step % 10000 == 0 and printsim == True:
				sim = similarity.eval()
				for i in xrange(valid_size):
					valid_word = reverse_dictionary[valid_examples[i]]
					top_k = 8  # number of nearest neighbors
					nearest = (-sim[i, :]).argsort()[1:top_k + 1]
					log_str = 'Nearest to %s:' % valid_word
					for k in xrange(top_k):
						close_word = reverse_dictionary[nearest[k]]
						log_str = '%s %s,' % (log_str, close_word)
					print(log_str)
		final_embeddings = normalized_embeddings.eval()

		# Save embedding result here
		emb_labels = [reverse_dictionary[i] for i in xrange(random_walks_size)]
		emb_df = pd.DataFrame(final_embeddings)
		emb_df.insert(0, 'labels', emb_labels)
		emb_df.to_csv(para_dict['emb_path'], index=False, sep=' ', header=None)

		# Write corresponding labels for the embeddings.
		with open(para_dict['log_dir'] + '/metadata.tsv', 'w') as f:
			for i in xrange(random_walks_size):
				f.write(reverse_dictionary[i] + '\n')

		# Save the model for checkpoints.
		saver.save(session, os.path.join(para_dict['log_dir'], 'model.ckpt'))

		# Create a configuration for visualizing embeddings with the labels in
		# TensorBoard.
		config = projector.ProjectorConfig()
		embedding_conf = config.embeddings.add()
		embedding_conf.tensor_name = embeddings.name
		embedding_conf.metadata_path = os.path.join(para_dict['log_dir'], 'metadata.tsv')
		projector.visualize_embeddings(writer, config)

	writer.close()

	# Step 6: Visualize the embeddings.

	# pylint: disable=missing-docstring
	# Function to draw visualization of distance between embeddings.
	def plot_with_labels(low_dim_embs, labels, savefig_path):
		assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
		plt.figure(figsize=(18, 18))  # in inches
		for i, label in enumerate(labels):
			x, y = low_dim_embs[i, :]
			plt.scatter(x, y)
			plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

		plt.savefig(savefig_path)

	try:
		# pylint: disable=g-import-not-at-top
		from sklearn.manifold import TSNE
		import matplotlib.pyplot as plt

		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
		plot_only = int(random_walks_size * para_dict['plot_pct'])
		low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
		labels = [reverse_dictionary[i] for i in xrange(plot_only)]
		plot_with_labels(low_dim_embs, labels, para_dict['plot_path']+'tsne.png')

	except ImportError as ex:
		print('Please install sklearn, matplotlib, and scipy to show embeddings.')
		print(ex)



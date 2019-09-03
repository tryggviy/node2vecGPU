# node2vecGPU
This is an implementation of node2vec with tensorflow, based on the original one of aditya-grover and word2vec tutorial of tensorflow. It can deal with graphs with massive number of nodes or densely connected graphs faster.

### Basic Usage
The usage is exactly the same as the original one, except for a few changes of hyper parameters for tensorflow.
##### Example
To run node2vec, execute the following command from the project home directory:
`python src/main.py --input ../graph/listed_weight.edgelist --output ../emb/karate.emb`

##### Options
You can check out the other options available to use with node2vec using:
`python src/main.py --help`

##### Input
The supported input format is an edgelist:

`node1_id_int node2_id_int <weight_float, optional>`

##### Output
The output is different from the original implementation.

The output file has n lines for a graph with n vertices.

The n lines are as follows:

`node_id dim1 dim2 ... dimd`
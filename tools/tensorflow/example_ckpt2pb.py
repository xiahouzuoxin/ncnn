import os, sys
import tensorflow as tf
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="convert tensorflow's checkpt to .pb")
    parser.add_argument('--restore', type=str,
            help='checkpoint prefix will be converted')
    parser.add_argument('--out', type=str,
            help='output .pb filename')
    return parser.parse_args()

args = get_args()
print(args)

n_inputs=26
n_actions=12

n_hidden_1=256
n_hidden_2=256

# Load parameters
g = tf.Graph()
with g.as_default():
    weight={
        'h1': tf.Variable(tf.random_normal([n_inputs,n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2,n_actions])),
    }
    bias={
        'h1':tf.Variable(tf.random_normal([n_hidden_1])),
        'h2':tf.Variable(tf.random_normal([n_hidden_2])),
        'out':tf.Variable(tf.random_normal([n_actions])),
    }

    stateInput=tf.placeholder('float',[None,n_inputs])
    layer1=tf.add(tf.matmul(stateInput,weight['h1']),bias['h1'])
    layer1=tf.nn.relu(layer1)
    layer2=tf.add(tf.matmul(layer1,weight['h2']),bias['h2'])
    layer2=tf.nn.relu(layer2)
    QValue=tf.add(tf.matmul(layer2,weight['out']),bias['out'])

    sess = tf.Session()

    restore_var = [ v for v in tf.global_variables() ]
    loader = tf.train.Saver(var_list=restore_var)
    loader.restore(sess, args.restore)

# Store variable
_W_h1=weight['h1'].eval(sess)
_W_h2=weight['h2'].eval(sess)
_W_out=weight['out'].eval(sess)
_b_h1=bias['h1'].eval(sess)
_b_h2=bias['h2'].eval(sess)
_b_out=bias['out'].eval(sess)

sess.close()

# Create new graph for exporting
g_2 = tf.Graph()
with g_2.as_default():
    # Reconstruct graph
    stateInput=tf.placeholder('float',[None,n_inputs])
    layer1=tf.add(tf.matmul(stateInput,_W_h1),_b_h1)
    layer1=tf.nn.relu(layer1)
    layer2=tf.add(tf.matmul(layer1,_W_h2),_b_h2)
    layer2=tf.nn.relu(layer2)
    QValue=tf.add(tf.matmul(layer2,_W_out),_b_out)

    sess_2 = tf.Session()
    init_2 = tf.initialize_all_variables();
    sess_2.run(init_2)

    graph_def = g_2.as_graph_def()
    
    tf.train.write_graph(graph_def, './', args.out, as_text=False)

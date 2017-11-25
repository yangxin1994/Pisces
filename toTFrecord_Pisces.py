#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np
import csv
import os
import time
import networkx as nx
import numba
import itertools
# ===========  global parameters  ===========

T = 5  # iteration
N = 2  # embedding_depth
D = 9  # dimensional
P = 64  # embedding_size
B = 10  # mini-batch
lr = 0.0001  # learning_rate
# MAX_SIZE = 0 # record the max number of a function's block
epochs = 100
is_debug = True

data_folder="./features/"
INDEX = data_folder + "index.csv"
PREFIX = ""
train_file = os.path.join(data_folder, "train"+PREFIX+".csv")
valid_file = os.path.join(data_folder, "vaild"+PREFIX+".csv")
test_file = os.path.join(data_folder, "test"+PREFIX+".csv")
TRAIN_TFRECORD="TFrecord/train_pisces_data_"+"100000"+PREFIX+".tfrecord"
TEST_TFRECORD="TFrecord/test_pisces_data_"+"100000"+PREFIX+".tfrecord"
VALID_TFRECORD="TFrecord/valid_pisces_data_"+"100000"+PREFIX+".tfrecord"

# ==================== load the function pairs list ===================
#       1.   load_dataset()      load the pairs list for learning, which are
#                                in train.csv, valid.csv, test.csv .
#       1-1. load_csv_as_pair()  process each csv file.
# =====================================================================

def load_dataset():
    """ load the pairs list for training, testing, validing
    """

    train_pair, train_label = load_csv_as_pair(train_file)
    valid_pair, valid_label = load_csv_as_pair(valid_file)
    test_pair, test_label = load_csv_as_pair(test_file)

    return train_pair, train_label, valid_pair, valid_label, test_pair, test_label


def load_csv_as_pair(pair_label_file):
    """ load each csv file, which record the pairs list for learning and its label ( 1 or -1 )
        csv file : uid, uid, 1/-1 eg: 1.1.128, 1.4.789, -1
        pair_dict = {(uid, uid) : -1/1}
    """
    pair_list = []
    label_list = []
    with open(pair_label_file, "r") as fp:
        pair_label = csv.reader(fp)
        for line in pair_label:
            pair_list.append([line[0], line[1]])
            label_list.append(int(line[2]))

    return pair_list, label_list


# ====================== load block info and cfg ======================
#       1.   load_all_data()     load the pairs list for learning, which are
#                                in train.csv, valid.csv, test.csv .
#       1-1. load_block_info()
#       1-2. load_graph()
# =====================================================================
def load_all_data():
    """ load all the real data, including blocks' featrue & functions' cfg using networkx
        uid_graph = {uid: nx_graph}
        feature_dict = {identifier : [[feature_vector]...]}, following the block orders
    """
    uid_graph = {}
    feature_dict = {}
    father  = {}
    child = {}

    # read the direcory list and its ID
    # traversal each record to load every folder's data
    with open(INDEX, "r") as fp:
        for line in csv.reader(fp):
            # index.csv : folder name, identifier
            # eg： openssl-1.0.1a_gcc_4.6_dir, 1.1

            # load_block_info: save all the blocks' feature vector into feature_dict;
            #                  return current file's block number saved into block_num;
            #                  return each function's block id list saved into cur_func_block_dict.
            block_uuid, cur_func_block_dict, func_uuid = load_block_info(os.path.join(data_folder, line[0], "block_info.csv"),
                                                             feature_dict, line[1])

            if is_debug:
                print "load cfg ..."
            # load every function's cfg
            load_graph(os.path.join(data_folder, line[0], "adj_info.txt"), block_uuid, cur_func_block_dict, line[1],
                       uid_graph)

            if is_debug:
                print "load call graph ..."
            load_call_graph(os.path.join(data_folder, line[0], "call_info.txt"), func_uuid, cur_func_block_dict, line[1],
                            feature_dict, father, child)


    return uid_graph, feature_dict, father, child


def load_block_info(feature_file, feature_dict, uid_prefix):
    """ load all the blocks' feature vector into feature dictionary.
        the two returned values are for next step, loading graph.
        return the block numbers —— using to add the single node of the graph
        return cur_func_blocks_dict —— using to generate every function's cfg（subgraph)
    """
    feature_dict[str(uid_prefix)] = []
    cur_func_blocks_dict = {}

    block_uuid = []
    func_uuid = []
    line_num = 0
    block_feature_dic = {}
    with open(feature_file, "r") as fp:
        if is_debug:
            print feature_file
        for line in csv.reader(fp):
            line_num += 1
            # skip the topic line
            #if line_num == 1:
            #    continue
            if line[0] == "":
                continue
            block_uuid.append(str(line[0]))
            # read every bolck's features
            block_feature = [float(x) for x in (line[4:13])]
            block_feature_dic.setdefault(str(line[0]),block_feature)

            # record each function's block id.
            # for next step to generate the control flow graph
            # so the root block need be add.
            if str(line[2]) not in cur_func_blocks_dict:
                func_uuid.append(str(line[0]))
                cur_func_blocks_dict[str(line[2])] = [str(line[0])]
            else:
                cur_func_blocks_dict[str(line[2])].append(str(line[0]))
        feature_dict[str(uid_prefix)].append(block_feature_dic)

    return block_uuid, cur_func_blocks_dict , func_uuid

def load_call_graph(graph_file, func_uuid, cur_func_blocks_dict, uid_prefix, feature_dict, father, child):

    graph = nx.read_edgelist(graph_file, create_using=nx.DiGraph(), nodetype=str)

    # add the missing vertexs which are not in edge_list
    for func_id in func_uuid:
        uid = uid_prefix + "." + str(func_id)
        if func_id not in graph.nodes():
            father.setdefault(uid,[0,0,0,0,0,0,0,0,0])
            child.setdefault(uid,[0,0,0,0,0,0,0,0,0])
        else:
            father_nodes = graph.predecessors(func_id)
            func_features = [0,0,0,0,0,0,0,0,0]
            for cur_func_node in father_nodes:
                if cur_func_blocks_dict.has_key(cur_func_node):
                    block_nodes = cur_func_blocks_dict[cur_func_node]
                    for cur_block_node in block_nodes:
                        cur_feature = feature_dict.get(str(uid_prefix))[0][cur_block_node]
                        for i in xrange(len(func_features)):
                            func_features[i] = func_features[i]+cur_feature[i]
            #print "father  ", uid, func_features
            father.setdefault(uid,func_features)

            child_nodes = graph.successors(func_id)
            func_features = [0,0,0,0,0,0,0,0,0]
            for cur_func_node in child_nodes:
                if cur_func_blocks_dict.has_key(cur_func_node):
                    block_nodes = cur_func_blocks_dict[cur_func_node]
                    for cur_block_node in block_nodes:
                        cur_feature = feature_dict.get(str(uid_prefix))[0][cur_block_node]
                        for i in xrange(len(func_features)):
                            func_features[i] = func_features[i]+cur_feature[i]
            #print "child  ", uid, func_features
            child.setdefault(uid,func_features)

def load_graph(graph_file, block_uuid, cur_func_blocks_dict, uid_prefix, uid_graph):
    """ load all the graph as networkx
    """
    graph = nx.read_edgelist(graph_file, create_using=nx.DiGraph(), nodetype=str)

    # add the missing vertexs which are not in edge_list
    for i in block_uuid:
        if i not in graph.nodes():
            graph.add_node(i)

    for func_id in cur_func_blocks_dict:
        graph_sub = graph.subgraph(cur_func_blocks_dict[func_id])
        uid = uid_prefix + "." + str(func_id)
        uid_graph[uid] = graph_sub


# =============== convert the real data to training data ==============
#       1.  construct_learning_dataset() combine the dataset list & real data
#       1-1. generate_adj_matrix_pairs()    traversal list and construct all the matrixs
#       1-1-1. convert_graph_to_adj_matrix()    process each cfg
#       1-2. generate_features_pair() traversal list and construct all functions' feature map
# =====================================================================
#@numba.jit
def construct_learning_dataset(uid_pair_list):
    """ Construct pairs dataset to train the model.
        attributes:
            adj_matrix_all  store each pairs functions' graph info, （i,j)=1 present i--》j, others （i,j)=0
            features_all    store each pairs functions' feature map
    """
    print "     start generate adj matrix pairs..."
    adj_matrix_all_1, adj_matrix_all_2 = generate_adj_matrix_pairs(uid_pair_list)

    call_father_1, call_father_2, call_child_1, call_child_2 = generate_call_features(uid_pair_list)

    print "     start generate features pairs..."
    ### !!! record the max number of a function's block
    features_all_1, features_all_2, max_size, num1, num2 = generate_features_pair(uid_pair_list)

    return adj_matrix_all_1, adj_matrix_all_2, features_all_1, features_all_2, num1, num2, max_size, call_father_1, call_father_2, call_child_1, call_child_2

#@numba.jit
def generate_call_features(uid_pair_list):
    father_1 = []
    father_2 = []
    child_1 = []
    child_2 = []
    # traversal all the pairs
    count = 0
    for uid_pair in uid_pair_list:
        if is_debug:
            count += 1
            print "         %04d call features, [ %s , %s ]"%(count, uid_pair[0], uid_pair[1])
        fea_vec=[0,0,0,0,0,0,0,0,0]
        if father.has_key(uid_pair[0]):
            fea_vec = father.get(uid_pair[0])
        print ",".join(str(i) for i in fea_vec)
        father_1.append(",".join(str(i) for i in fea_vec))

        fea_vec=[0,0,0,0,0,0,0,0,0]
        if child.has_key(uid_pair[0]):
            fea_vec = child.get(uid_pair[0])
        print ",".join(str(i) for i in fea_vec)
        child_1.append(",".join(str(i) for i in fea_vec))

        fea_vec=[0,0,0,0,0,0,0,0,0]
        if father.has_key(uid_pair[1]):
            fea_vec = father.get(uid_pair[1])
        print ",".join(str(i) for i in fea_vec)
        father_2.append(",".join(str(i) for i in fea_vec))

        fea_vec=[0,0,0,0,0,0,0,0,0]
        if child.has_key(uid_pair[1]):
            fea_vec = child.get(uid_pair[1])
        print ",".join(str(i) for i in fea_vec)
        child_2.append(",".join(str(i) for i in fea_vec))

    return father_1, father_2, child_1, child_2


def generate_adj_matrix_pairs(uid_pair_list):
    """ construct all the function pairs' cfg matrix.
    """
    adj_matrix_all_1 = []
    adj_matrix_all_2 = []
    # traversal all the pairs
    count = 0
    for uid_pair in uid_pair_list:
        if is_debug:
            count += 1
            print "         %04d martix, [ %s , %s ]"%(count, uid_pair[0], uid_pair[1])
        adj_matrix_pair = []
        # each pair process two function
        graph = uid_graph[uid_pair[0]]
        # origion_adj_1 = np.array(nx.convert_matrix.to_numpy_matrix(graph, dtype=float))
        # origion_adj_1.resize(size, size, refcheck=False)
        # adj_matrix_all_1.append(origion_adj_1.tolist())
        adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph, dtype=float))
        adj_str = adj_arr.astype(np.string_)
        adj_matrix_all_1.append(",".join(list(itertools.chain.from_iterable(adj_str))))

        graph = uid_graph[uid_pair[1]]
        # origion_adj_2 = np.array(nx.convert_matrix.to_numpy_matrix(graph, dtype=float))
        # origion_adj_2.resize(size, size, refcheck=False)
        # adj_matrix_all_2.append(origion_adj_2.tolist())
        adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph, dtype=float))
        adj_str = adj_arr.astype(np.string_)
        adj_matrix_all_2.append(",".join(list(itertools.chain.from_iterable(adj_str))))

    return adj_matrix_all_1, adj_matrix_all_2


#@numba.jit
def convert_graph_to_adj_matrix(graph):
    """ convert the control flow graph as networkx to a adj matrix （v_num * v_num).
        1 present an edge; 0 present no edge
    """
    node_list = graph.nodes()
    adj_matrix = []

    # get all the block id in the cfg
    # construct a v_num * v_num adj martix
    for u in node_list:
        # traversal each block's edgd list,to add the
        u_n = graph.neighbors(u)
        neighbors = []
        for tmp in u_n:
            neighbors.append(tmp)
        node_adj = []
        for v in node_list:
            if v in neighbors:
                node_adj.append(1)
            else:
                node_adj.append(0)
        adj_matrix.append(node_adj)
    # print adj_matrix
    return adj_matrix


#@numba.jit
def generate_features_pair(uid_pair_list):
    """ Construct each function pairs' block feature map.
    """
    node_vector_all_1 = []
    node_vector_all_2 = []
    num1 = []
    num2 = []
    node_length = []
    # traversal all the pairs
    count = 0
    for uid_pair in uid_pair_list:
        if is_debug:
            count += 1
            print "         %04d feature, [ %s , %s ]"%(count, uid_pair[0], uid_pair[1])
        node_vector_pair = []
        # each pair process two function
        uid = uid_pair[0]
        node_list = uid_graph[uid].nodes()
        uid_prefix = uid.rsplit('.', 1)[0]  # 从右边第一个'.'分界，分成两个字符串 即 identifier, function_id
        node_vector = []
        for node in node_list:
            node_vector.append(feature_dict[str(uid_prefix)][0][node])
        node_length.append(len(node_vector))
        num1.append(len(node_vector))
        node_arr = np.array(node_vector)
        node_str = node_arr.astype(np.string_)
        node_vector_all_1.append(",".join(list(itertools.chain.from_iterable(node_str))))

        uid = uid_pair[1]
        node_list = uid_graph[uid].nodes()
        uid_prefix = uid.rsplit('.', 1)[0]  # 从右边第一个'.'分界，分成两个字符串 即 identifier, function_id
        node_vector = []
        for node in node_list:
            node_vector.append(feature_dict[str(uid_prefix)][0][node])
        node_length.append(len(node_vector))
        num2.append(len(node_vector))
        node_arr = np.array(node_vector)
        node_str = node_arr.astype(np.string_)
        node_vector_all_2.append(",".join(list(itertools.chain.from_iterable(node_str))))

    num1_re = np.array(num1)
    num2_re = np.array(num2)
    #num1_re = num1_arr.astype(np.string_)
    #num2_re = num2_arr.astype(np.string_)
    return node_vector_all_1, node_vector_all_2, np.max(node_length),num1_re,num2_re

    # =============== convert the real data to training data ==============
    #       1.  construct_learning_dataset() combine the dataset list & real data
    #       1-1. generate_adj_matrix_pairs()    traversal list and construct all the matrixs
    #       1-1-1. convert_graph_to_adj_matrix()    process each cfg
    #       1-2. generate_features_pair() traversal list and construct all functions' feature map
    # =====================================================================
    """ Parameter P = 64, D = 8, T = 7, N = 2,                  B = 10
         X_v = D * 1   <--->   8 * v_num * 10
         W_1 = P * D   <--->   64* 8    W_1 * X_v = 64*1
        mu_0 = P * 1   <--->   64* 1
         P_1 = P * P   <--->   64*64
         P_2 = P * P   <--->   64*64
        mu_2/3/4/5 = P * P     <--->  64*1
        W_2 = P * P     <--->  64*64
    """

def structure2vec(mu_prev, adj_matrix, x, name="structure2vec"):
    """ Construct pairs dataset to train the model.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # n层全连接层 + n-1层激活层
        # n层全连接层  将v_num个P*1的特征汇总成P*P的feature map
        # 初始化P1,P2参数矩阵，截取的正态分布模式初始化  stddev是用于初始化的标准差
        # 合理的初始化会给网络一个比较好的训练起点，帮助逃脱局部极小值（or 鞍点）
        W_1 = tf.get_variable('W_1', [D, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        P_1 = tf.get_variable('P_1', [P, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        P_2 = tf.get_variable('P_2', [P, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        L = tf.reshape(tf.matmul(adj_matrix, mu_prev, transpose_a=True), (-1, P))  # v_num * P
        S = tf.reshape(tf.matmul(tf.nn.relu(tf.matmul(L, P_2)), P_1), (-1, P))

        return tf.tanh(tf.add(tf.reshape(tf.matmul(tf.reshape(x, (-1, D)), W_1), (-1, P)), S))

def structure2vec_net(adj_matrix, x, v_num):
    with tf.variable_scope("structure2vec_net", reuse=tf.AUTO_REUSE):
        B_mu_5 = tf.Variable(tf.zeros(shape = [0, P]), trainable=False)
        for i in range(B):
            cur_size = tf.to_int32(v_num[i][0])
            # test = tf.slice(B_mu_0[i], [0, 0], [cur_size, P])
            mu_0 = tf.reshape(tf.zeros(shape = [cur_size, P]),(cur_size,P))
            adj = tf.slice(adj_matrix[i], [0, 0], [cur_size, cur_size])
            fea = tf.slice(x[i],[0,0], [cur_size,D])
            mu_1 = structure2vec(mu_0, adj, fea)  # , name = 'mu_1')
            mu_2 = structure2vec(mu_1, adj, fea)  # , name = 'mu_2')
            mu_3 = structure2vec(mu_2, adj, fea)  # , name = 'mu_3')
            mu_4 = structure2vec(mu_3, adj, fea)  # , name = 'mu_4')
            mu_5 = structure2vec(mu_4, adj, fea)  # , name = 'mu_5')

            w_2 = tf.get_variable('w_2', [P, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            # B_mu_5.append(tf.matmul(tf.reshape(tf.reduce_sum(mu_5, 0), (1, P)), w_2))
            B_mu_5 = tf.concat([B_mu_5,tf.matmul(tf.reshape(tf.reduce_sum(mu_5, 0), (1, P)), w_2)],0)

        return B_mu_5

def contrastive_loss(labels, distance):
    #    tmp= y * tf.square(d)
    #    #tmp= tf.mul(y,tf.square(d))
    #    tmp2 = (1-y) * tf.square(tf.maximum((1 - d),0))
    #    return tf.reduce_sum(tmp +tmp2)/B/2
    #    print "contrastive_loss", y,
    loss = tf.to_float(tf.reduce_sum(tf.square(distance - labels)))
    return loss


def compute_accuracy(prediction, labels):
    accu = 0.0
    threshold = 0.5
    for i in xrange(len(prediction)):
        if labels[i][0] == 1:
            if prediction[i][0] > threshold:
                accu += 1.0
        else:
            if prediction[i][0] < threshold:
                accu += 1.0
    acc = accu / len(prediction)
    return acc


#    return labels[prediction.ravel() < 0.5].mean()
# return tf.reduce_mean(labels[prediction.ravel() < 0.5])

def cal_distance(model1, model2):
    a_b = tf.reduce_sum(tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(model1,(1,-1)),
                                                             tf.reshape(model2,(1,-1))],0),0),(B,P)),1,keep_dims=True)
    a_norm = tf.sqrt(tf.reduce_sum(tf.square(model1),1,keep_dims=True))
    b_norm = tf.sqrt(tf.reduce_sum(tf.square(model2),1,keep_dims=True))
    distance = a_b/tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(a_norm,(1,-1)),
                                                        tf.reshape(b_norm,(1,-1))],0),0),(B,1))
    return distance


def next_batch(s,e,adj_matrix_train_1, adj_matrix_train_2, node_vector_train_1, node_vector_train_2, label_list_train):

    graph_1eft = adj_matrix_train_1[s:e]
    feature_1eft = node_vector_train_1[s:e]
    graph_right = adj_matrix_train_2[s: e]
    feature_right = node_vector_train_2[s:e]

    label = np.reshape(label_list_train[s:e], [B, 1])

    return graph_1eft, feature_1eft, graph_right, feature_right, label


# ========================== the main function ========================
#       1.  load_dataset()  load the train, valid, test csv file.
#       2.  load_all_data() load the origion data, including block info, cfg by networkx.
#       3.  construct_learning_dataset() combine the csv file and real data, construct training dataset.
# =====================================================================
# 1. load the train, valid, test csv file.
data_time = time.time()
train_pair, train_label, valid_pair, valid_label, test_pair, test_label = load_dataset()
print "1. loading pairs list time", time.time() - data_time, "(s)"

# 2. load the origion data, including block info, cfg by networkx.
graph_time = time.time()
uid_graph, feature_dict, father, child = load_all_data()
print "2. loading graph data time", time.time() - graph_time, "(s)"

# 3. construct training dataset.
cons_time = time.time()

# ======================= construct valid data =====================
valid_adj_matrix_1, valid_adj_matrix_2, valid_feature_map_1, valid_feature_map_2, valid_num1, valid_num2, valid_max, \
valid_call_father_1, valid_call_father_2, valid_call_child_1, valid_call_child_2 = construct_learning_dataset(valid_pair)
# ========================== store in pickle ========================
node_list = np.linspace(valid_max,valid_max, len(valid_label),dtype=int)
writer = tf.python_io.TFRecordWriter(VALID_TFRECORD)
for item1,item2,item3,item4,item5,item6, item7, item8, item9, item10, item11, item12  in itertools.izip(
        valid_label, valid_adj_matrix_1, valid_adj_matrix_2, valid_feature_map_1, valid_feature_map_2, valid_num1, valid_num2,
        node_list,valid_call_father_1, valid_call_father_2, valid_call_child_1, valid_call_child_2):
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {'label':tf.train.Feature(int64_list = tf.train.Int64List(value=[item1])),
                       'adj_matrix_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item2])),
                       'adj_matrix_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item3])),
                       'feature_map_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item4])),
                       'feature_map_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item5])),
                       'num1':tf.train.Feature(int64_list = tf.train.Int64List(value=[item6])),
                       'num2':tf.train.Feature(int64_list = tf.train.Int64List(value=[item7])),
                       'max': tf.train.Feature(int64_list=tf.train.Int64List(value=[item8])),
                       'father_1':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item9])),
                       'father_2':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item10])),
                       'child_1':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item11])),
                       'child_2':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item12]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
writer.close()
# ========================== clean memory ========================
# del valid_pair, valid_adj_matrix_1,valid_adj_matrix_2,valid_feature_map_1,valid_feature_map_2,valid_max

# ======================= construct train data =====================
train_adj_matrix_1, train_adj_matrix_2, train_feature_map_1, train_feature_map_2, train_num1, train_num2, train_max, \
train_call_father_1, train_call_father_2, train_call_child_1, train_call_child_2 = construct_learning_dataset(train_pair)
# ========================== store in pickle ========================
node_list = np.linspace(train_max,train_max, len(train_label),dtype=int)
writer = tf.python_io.TFRecordWriter(TRAIN_TFRECORD)
for item1,item2,item3,item4,item5,item6, item7, item8, item9, item10, item11, item12 in itertools.izip(
        train_label, train_adj_matrix_1, train_adj_matrix_2, train_feature_map_1, train_feature_map_2, train_num1, train_num2,
        node_list,  train_call_father_1, train_call_father_2, train_call_child_1, train_call_child_2 ):
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {'label':tf.train.Feature(int64_list = tf.train.Int64List(value=[item1])),
                       'adj_matrix_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item2])),
                       'adj_matrix_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item3])),
                       'feature_map_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item4])),
                       'feature_map_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item5])),
                       'num1':tf.train.Feature(int64_list = tf.train.Int64List(value=[item6])),
                       'num2':tf.train.Feature(int64_list = tf.train.Int64List(value=[item7])),
                       'max': tf.train.Feature(int64_list=tf.train.Int64List(value=[item8])),
                       'father_1':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item9])),
                       'father_2':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item10])),
                       'child_1':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item11])),
                       'child_2':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item12]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
writer.close()
# ========================== clean memory ========================
# del train_pair, train_adj_matrix_1,train_adj_matrix_2,train_feature_map_1,train_feature_map_2,train_max

# ======================= construct test data =====================
test_adj_matrix_1, test_adj_matrix_2, test_feature_map_1, test_feature_map_2,test_num1, test_num2, test_max, \
test_call_father_1, test_call_father_2, test_call_child_1, test_call_child_2 = construct_learning_dataset(test_pair)
# ========================== store in pickle ========================
node_list = np.linspace(test_max,test_max, len(test_label),dtype=int)
writer = tf.python_io.TFRecordWriter(TEST_TFRECORD)
for item1,item2,item3,item4,item5,item6, item7, item8 , item9, item10, item11, item12 in itertools.izip(
        test_label, test_adj_matrix_1, test_adj_matrix_2, test_feature_map_1, test_feature_map_2,test_num1, test_num2,
        node_list, test_call_father_1, test_call_father_2, test_call_child_1, test_call_child_2 ):
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {'label':tf.train.Feature(int64_list = tf.train.Int64List(value=[item1])),
                       'adj_matrix_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item2])),
                       'adj_matrix_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item3])),
                       'feature_map_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item4])),
                       'feature_map_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item5])),
                       'num1':tf.train.Feature(int64_list = tf.train.Int64List(value=[item6])),
                       'num2':tf.train.Feature(int64_list = tf.train.Int64List(value=[item7])),
                       'max': tf.train.Feature(int64_list=tf.train.Int64List(value=[item8])),
                       'father_1':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item9])),
                       'father_2':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item10])),
                       'child_1':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item11])),
                       'child_2':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item12]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
writer.close()

import numpy as np
import pickle, struct, socket, math

def get_even_odd_from_one_hot_label(label):
    for i in range(0, len(label)):
        if label[i] == 1:
            c = i % 2
            if c == 0:
                c = 1
            elif c == 1:
                c = -1
            return c


def get_index_from_one_hot_label(label):
    for i in range(0, len(label)):
        if label[i] == 1:
            return [i]


def get_one_hot_from_label_index(label, number_of_labels=10):
    one_hot = np.zeros(number_of_labels)
    one_hot[label] = 1
    return one_hot


def send_msg(sock, msg):
    #print(msg[0], 'sent to', sock.getpeername())
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)



def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        #print(msg)
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg


def moving_average(param_mvavr, param_new, movingAverageHoldingParam):
    if param_mvavr is None or np.isnan(param_mvavr):
        param_mvavr = param_new
    else:
        if not np.isnan(param_new):
            param_mvavr = movingAverageHoldingParam * param_mvavr + (1 - movingAverageHoldingParam) * param_new
    return param_mvavr


def get_indices_each_node_case(n_nodes, maxCase, label_list) -> object:
    indices_each_node_case = []

    for i in range(0, maxCase):
        indices_each_node_case.append([])

    for i in range(0, n_nodes):
        for j in range(0, maxCase):
            indices_each_node_case[j].append([])

    # indices_each_node_case is a big list that contains N-number of sublists. Sublist n contains the indices that should be assigned to node n

    min_label = min(label_list)
    max_label = max(label_list)
    num_labels = max_label - min_label + 1

    label_record = [0]*10

    for i in range(0, len(label_list)):
        label_record[label_list[i]] += 1
        # case 1
        indices_each_node_case[0][(i % n_nodes)].append(i)

        # case 2
        tmp_target_node = int((label_list[i] - min_label) % n_nodes)
        if n_nodes > num_labels:
            tmp_min_index = 0
            tmp_min_val = math.inf
            for n in range(0, n_nodes):
                if n % num_labels == tmp_target_node and len(indices_each_node_case[1][n]) < tmp_min_val:
                    tmp_min_val = len(indices_each_node_case[1][n])
                    tmp_min_index = n
            tmp_target_node = tmp_min_index
        indices_each_node_case[1][tmp_target_node].append(i)

        # case 3
        tmp = int(np.ceil(min(n_nodes, num_labels) / 2))
        if label_list[i] < (min_label + max_label) / 2:
            tmp_target_node = i % tmp
        elif n_nodes > 1:
            tmp_target_node = int(((label_list[i] - min_label) % (min(n_nodes, num_labels) - tmp)) + tmp)

        if n_nodes > num_labels:
            tmp_min_index = 0
            tmp_min_val = math.inf
            for n in range(0, n_nodes):
                if n % num_labels == tmp_target_node and len(indices_each_node_case[2][n]) < tmp_min_val:
                    tmp_min_val = len(indices_each_node_case[2][n])
                    tmp_min_index = n
            tmp_target_node = tmp_min_index

        indices_each_node_case[2][tmp_target_node].append(i)

    return indices_each_node_case

def clip_function(grad, weight_dimensions, clip_value, sigma):
    for i in range(len(weight_dimensions)):
        if i == 0:
            current_norm = np.linalg.norm(grad[:weight_dimensions[i]])
            if current_norm > clip_value[i]:
                grad[:weight_dimensions[i]] = grad[:weight_dimensions[i]] / current_norm * clip_value[i]
            grad[:weight_dimensions[i]] += 1/weight_dimensions[i] *np.random.normal(loc=0, scale=(clip_value[i]*sigma), size=weight_dimensions[i])
        else:
            current_norm = np.linalg.norm(grad[weight_dimensions[i-1]:weight_dimensions[i]])
            if current_norm > clip_value[i]:
                grad[weight_dimensions[i-1]:weight_dimensions[i]] = grad[weight_dimensions[i-1]:weight_dimensions[i]] / current_norm * clip_value[i]
            grad[weight_dimensions[i - 1]:weight_dimensions[i]] += 1/(weight_dimensions[i]-weight_dimensions[i-1]) *np.random.normal(loc=0, scale=(clip_value[i]*sigma), size=weight_dimensions[i]-weight_dimensions[i-1])
    return grad

def add_global_noise(w_set, weight_dimensions, sigma, data_size_local_all, data_size_total, w_global_prev):

    w_global = np.zeros_like(w_set[0])
    clip_bound = np.array([2.80, 0.45, 12.46, 0.68, 68.32, 1.60, 5.22, 0.32]) # sensitivity defined for MNIST by some empirical experiments
    w_num = len(w_set)

    for i in range(w_num):
        w_set[i] = clip_function(w_set[i], weight_dimensions, clip_bound, sigma)

    for i in range(w_num):
        w_global += w_set[i]

    w_global = w_global / w_num

    w_global = clip_function(w_global, weight_dimensions, clip_bound, 0)

    return w_global
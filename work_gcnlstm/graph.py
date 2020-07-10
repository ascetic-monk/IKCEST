import numpy as np
import matplotlib.pyplot as plt

def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    """Load weight matrix function."""
    try:
        W = np.load(file_path)
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 100.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (
            np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W


def gen_A(A):
    import pickle
    # result = pickle.load(open(adj_file, 'rb'))
    # _adj = result['adj']
    # _nums = result['nums']
    # _nums = _nums[:, np.newaxis]
    # _nums = (A/np.sum(A, axis=1))[:, np.newaxis]
    # _adj = A / _nums
    _adj = A
    # _adj[_adj < t] = 0
    # _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(A.shape[0], np.int)
    return _adj


def graph_generate(adj_path='../../proj_baiduAI/dataset/data_processed_all/adj_matrix.npy'):
    W = weight_matrix(adj_path, scaling=False)
    plt.hist(np.reshape(1 - np.exp(-W), (-1,)))
    region_nums = [118, 30, 135, 75, 34, 331, 38, 53, 33, 8, 48]
    idx = 0
    rate = 0.01
    for region_num in region_nums:
        W_sub = W[idx:idx+region_num, idx:idx+region_num]
        W_total = W_sub.shape[0] * W_sub.shape[1]
        thr = np.sort(np.reshape(W_sub, (-1,)))[-int(rate * W_total)]
        W[idx:idx+region_num, idx:idx+region_num] = (W_sub > thr) # * np.log(W_sub+1e-1)
        idx += region_num
    adj = gen_A(W)
    return adj


if __name__ == '__main__':
    W = weight_matrix('../../proj_baiduAI/dataset/data_processed_all/adj_matrix.npy', scaling=False)
    plt.hist(np.reshape(1 - np.exp(-W), (-1,)))
    region_nums = [118, 30, 135, 75, 34, 331, 38, 53, 33, 8, 48]
    idx = 0
    rate = 0.01
    for region_num in region_nums:
        W_sub = W[idx:idx+region_num, idx:idx+region_num]
        W_total = W_sub.shape[0] * W_sub.shape[1]
        thr = np.sort(np.reshape(W_sub, (-1,)))[-int(rate * W_total)]
        W[idx:idx+region_num, idx:idx+region_num] = (W_sub > thr)# * np.log(W_sub+1e-1)
        idx += region_num
    adj = gen_A(W)


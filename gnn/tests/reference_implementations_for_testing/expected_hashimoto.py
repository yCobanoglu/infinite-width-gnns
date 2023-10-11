# from https://github.com/zhengdao-chen/GNN4CD
import numpy as np


def get_Pm(W):
    N = W.shape[0]
    assert np.allclose(W, W.T)
    W_ = W * (np.ones([N, N]) - np.eye(N))
    assert np.allclose(W, W_)
    M = int(W_.sum()) // 2
    p = 0
    Pm = np.zeros([N, M * 2])
    for n in range(N):
        for m in range(n + 1, N):
            if W_[n][m] == 1:
                Pm[n][p] = 1
                Pm[m][p] = 1
                Pm[n][p + M] = 1
                Pm[m][p + M] = 1
                p += 1
    assert np.allclose(Pm[:, :M], Pm[:, M:])
    return Pm


def get_Pd(W):
    N = W.shape[0]
    W = W * (np.ones([N, N]) - np.eye(N))
    M = int(W.sum()) // 2
    p = 0
    Pd = np.zeros([N, M * 2])
    for n in range(N):
        for m in range(n + 1, N):
            if W[n][m] == 1:
                Pd[n][p] = 1
                Pd[m][p] = -1
                Pd[n][p + M] = -1
                Pd[m][p + M] = 1
                p += 1
    return Pd


def get_P(W):
    P = np.concatenate((np.expand_dims(get_Pm(W), 2), np.expand_dims(get_Pd(W), 2)), axis=2)
    return P


def get_NB_2(W):
    Pm = get_Pm(W)
    Pd = get_Pd(W)
    Pf = (Pm + Pd) / 2
    Pt = (Pm - Pd) / 2
    NB = np.transpose(Pt).dot(Pf) * (1 - np.transpose(Pf).dot(Pt))
    return NB

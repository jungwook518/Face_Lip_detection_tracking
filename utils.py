import numpy as np

def _calc_cent_pts(bbox):
    centers = []
    if np.shape(bbox) == (4,):
        bbox = [bbox]
    
    for idx, box in enumerate(bbox):
        if box is None:
            center_x = np.inf
            center_y = np.inf
        else:
            center_x = int(box[0] + box[2]/2)
            center_y = int(box[1] + box[3]/2)
        centers.append([center_x, center_y])
    return centers

def _l2_dist(a, b):
    if a == np.inf or b == np.inf:
        return np.inf
    return np.linalg.norm(np.asarray(a) - np.asarray(b))

def _l2_dist_qk(q, k):
    if np.shape(k) == (2,):
        k = [k]

    dist = np.zeros(len(k))
    for idx in range(len(k)):
        dist[idx] = _l2_dist(q, k[idx])
    
    return dist

def _select_q2k(query, keys):
    cent_q = _calc_cent_pts(query)
    cent_k = _calc_cent_pts(keys)
    dist_qk = _l2_dist_qk(cent_q, cent_k)
    min_idx = np.where(dist_qk == np.min(dist_qk))
    
    return min_idx[0][0], np.min(dist_qk)

def _check_dupl(listOfElems):
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True
    

def find_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    out = data[s>m]
    return out[out>np.mean(data)]

def find_q2k(queries, keys):
    '''

    queries     : [[x0, y0] ... [xn, yn]]
    keys        : [[x0, y0] ... [xm, ym]]
    qk_list     : [[idx_0, dist_0] ... [idx_n, dist_n]]
    
    '''
    qk_list = []
    k_idx_list = []
    for i, query in enumerate(queries):
        k_idx, k_val = _select_q2k(query, keys)
        qk_list.append([k_idx, k_val])
        k_idx_list.append(k_idx)

    return qk_list

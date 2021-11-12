import numpy as np
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def GaussianPDF_1D(mu, sigma, length):
    # create an array
    half_len = length / 2

    if np.remainder(length, 2) == 0:
        ax = np.arange(-half_len, half_len, 1)
    else:
        ax = np.arange(-half_len, half_len + 1, 1)

    ax = ax.reshape([-1, ax.size])
    denominator = sigma * np.sqrt(2 * np.pi)
    nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )

    return nominator / denominator

def GaussianPDF_2D(mu, sigma, row, col):
    # create row vector as 1D Gaussian pdf
    g_row = GaussianPDF_1D(mu, sigma, row)
    # create column vector as 1D Gaussian pdf
    g_col = GaussianPDF_1D(mu, sigma, col).transpose()

    return signal.convolve2d(g_row, g_col, 'full')

def rgb2gray(I_rgb):
    r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
    I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return I_gray

def interp2(v, xq, yq):

	if len(xq.shape) == 2 or len(yq.shape) == 2:
		dim_input = 2
		q_h = xq.shape[0]
		q_w = xq.shape[1]
		xq = xq.flatten()
		yq = yq.flatten()

	h = v.shape[0]
	w = v.shape[1]
	if xq.shape != yq.shape:
		raise 'query coordinates Xq Yq should have same shape'


	x_floor = np.floor(xq).astype(np.int32)
	y_floor = np.floor(yq).astype(np.int32)
	x_ceil = np.ceil(xq).astype(np.int32)
	y_ceil = np.ceil(yq).astype(np.int32)

	x_floor[x_floor<0] = 0
	y_floor[y_floor<0] = 0
	x_ceil[x_ceil<0] = 0
	y_ceil[y_ceil<0] = 0

	x_floor[x_floor>=w-1] = w-1
	y_floor[y_floor>=h-1] = h-1
	x_ceil[x_ceil>=w-1] = w-1
	y_ceil[y_ceil>=h-1] = h-1

	v1 = v[y_floor, x_floor]
	v2 = v[y_floor, x_ceil]
	v3 = v[y_ceil, x_floor]
	v4 = v[y_ceil, x_ceil]

	lh = yq - y_floor
	lw = xq - x_floor
	hh = 1 - lh
	hw = 1 - lw

	w1 = hh * hw
	w2 = hh * lw
	w3 = lh * hw
	w4 = lh * lw

	interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

	if dim_input == 2:
		return interp_val.reshape(q_h,q_w)
	return interp_val
    
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

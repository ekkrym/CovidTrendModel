# ============================================================================
# import modules
# ============================================================================
import numpy as np
from sklearn.linear_model import HuberRegressor
from scipy.linalg import toeplitz, svd


# ============================================================================
# functions
# ============================================================================
def smooth_data(y, W=5):
    # robust linear regressor
    huber = HuberRegressor()
    # data dimensions
    ysmo = np.zeros(y.size)
    c = np.zeros(y.size)
    # time index
    t = np.arange(y.size).reshape((-1, 1))

    # loop over time
    for kk in np.arange(y.size - W + 1):
        # fit predictor
        huber.fit(t[kk:kk + W, :], y[kk:kk + W])
        # smoothed objective
        ysmo[kk:kk + W] += huber.predict(t[kk:kk + W].reshape((-1, 1)))
        c[kk:kk + W] += 1
    # return averaged prediction
    return ysmo / c


def prepare_data(y, method):
    if method == 'AR(1) rate':
        d = np.log(y[1:] / y[:-1] - 1)
    else:
        d = y.copy()
        if 'day' in method:
            d = np.diff(d)
        if 'log' in method:
            d = np.log(d+1)

    # return data
    return d


def linear_model_fit_predict(y, smoothed_dat=[],
                             method='AR(1) rate', weighting=None,
                             p=1, W=2, H=7, Ws=11):
    # linear regressor (Huber loss)
    huber = HuberRegressor()

    # data preparation
    # ------------------------------------------------------------------------
    # basic smoothing via convolution with triangular window
    if len(smoothed_dat) == 0:
        smoothed_dat = np.convolve(np.pad(np.array(y).astype(np.float), (1, 1), 'edge'), [0.25, 0.5, 0.25],
                                   mode='valid')
    # smooth data in log space
    s = smooth_data(np.log(smoothed_dat), W=Ws)
 
    s = np.exp(smooth_data(np.pad(s, (0, 2), 'edge'), W=7)[:-2])
 
    d = prepare_data(s, method) 
    # feature and target values
    # ------------------------------------------------------------------------
    # target values
    ytrain = d[-W:]

    # features
    M = (ytrain.size) // 2
    if np.mod(ytrain.size, 2) == 0:
        x = np.arange(-M, M);
        m = M
    else:
        x = np.arange(-M, M + 1);
        m = M + 1
    # vandermonde matrix for polynomial fitting
    Xtrain = np.fliplr(np.vander(x, p + 1))[:, 1:]
    # weights
    weight = np.ones(ytrain.size)  # np.sqrt(d[-L-H+1:-H])

    # fit model
    # ------------------------------------------------------------------------
    huber.fit(Xtrain, ytrain)

    # prediction
    # ------------------------------------------------------------------------
    if method == 'AR(1) rate':

        a = np.exp(huber.coef_)
        rho = np.exp(huber.intercept_) * a ** (W - 1)
        z = np.zeros(H)
        z[0] = (1 + rho) * smoothed_dat[-1]
        for jj in np.arange(1, H):
            z[jj] = (1 + rho * a ** jj) * z[jj - 1]

    else:

        # prediction
        z = huber.predict(np.fliplr(np.vander(np.arange(m, m + H), p + 1))[:, 1:])

        # convert to linear scale
        if 'log' in method:
            z = np.exp(z)

        # convert to cumulative cases
        if 'day' in method:
            z = np.cumsum(z) + smoothed_dat[-1]

    # predicted cumulative cases (include last observation)
    yp = np.zeros(H + 1);
    yp[0] = smoothed_dat[-1]
    yp[1:] = z

    # return prediction
    return yp


def sum_diags(A):
    """
    Sum the diagonals of a matrix.
    """
    # get dimensions
    m, n = A.shape

    # create output variable
    d = np.zeros(m + n - 1).astype(A.dtype)

    # loop over diagonals
    for ii in range(m):
        d[ii] = np.sum([A[m - 1 - ii + j, j] for j in range(np.min([n, ii + 1]))])
    for ii in range(n - 1):
        d[m + ii] = np.sum([A[j, j + 1 + ii] for j in range(np.min([m, n - ii - 1]))])

    # return sum
    return d


def ARfit(y, K=1, numiter=10):
    """
    AR model fitting using Cadzow denoising + Prony's method.
    """
    # variables
    Y = y.copy()
    L = int(np.ceil(Y.size / 2))
    w = sum_diags(np.ones((Y.size - L, L + 1)))

    # alternate projections
    for kk in range(numiter):
        # rank-K projection
        U, s, Vh = svd(toeplitz(Y[L:], Y[np.arange(L, -1, -1)]))
        R = s[0] * np.outer(U[:, 0], Vh[0, :])
        for jj in np.arange(1, K):
            R += s[jj] * np.outer(U[:, jj], Vh[jj, :])
        # projection onto toeplitz
        Y = np.flipud(sum_diags(R) / w)

    # find annihilating filter
    Qh = svd(toeplitz(Y[K:], Y[np.arange(K, -1, -1)]))[2]
    h = Qh[-1, :].conj()
    h = h / h[0]

    # return coefficients and smoothed data
    return h, Y


def ARpredict(y, h, H=7):
    """
    AR model prediction.
    """

    # AR coefficients
    v = -np.flipud(h[1:])

    # order of the model
    p = len(h) - 1

    # prediction
    z = np.zeros(H + p)
    z[:p] = y[-p:]
    for ii in range(H):
        z[ii + p] = np.sum(z[ii:ii + p] * v)

    return z[-H:]


def ar_model_fit_predict(y, smoothed_dat=[], method='AR(1) log day', p=1, W=-1, H=7, Ws=11):
    # data preparation
    # ------------------------------------------------------------------------
    # basic smoothing via convolution with triangular window
    if len(smoothed_dat) == 0:
        smoothed_dat = np.convolve(np.pad(np.array(y).astype(np.float), (1, 1), 'edge'), [0.25, 0.5, 0.25],
                                   mode='valid')
    # data smoothing
    d = np.exp(smooth_data(np.log(smoothed_dat), W=Ws))
    # data transformation
    if 'day' in method: d = np.diff(d)
    if 'log' in method: d = np.log(d)

    # fit AR(p) model
    # ------------------------------------------------------------------------
    if W < 0:
        h, dhat = ARfit(d, K=p)  # all history
    else:
        h, dhat = ARfit(d[-W:], K=p)  # recent history

    # prediction
    # ------------------------------------------------------------------------
    z = ARpredict(dhat, h, H=H)

    # predicted cumulative counts
    yp = np.zeros(H + 1);
    yp[0] = dhat[-1];
    yp[1:] = z

    # undo transformation
    if 'log' in method: yp[1:] = np.exp(z)
    if 'day' in method: yp = np.cumsum(yp)

    # return prediction
    return yp

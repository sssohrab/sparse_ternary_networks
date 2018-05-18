import numpy as np
from itertools import combinations
import scipy.special as sps
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import scipy.linalg as spla
###########################################3
def keep_k_largest_mags(inp, k, axis=0):
    """
    A function to keep only the k elements with the largest magnitudes in an array.

    axis = 0 chooses the largest magnitudes among columns while axis = 1 chooses among rows.
    """
    k = int(k)
    out = np.zeros(np.shape(inp))
    if axis == 0:
        if k < np.shape(inp)[0]:
            sorted_col_idx = np.argpartition(-np.abs(inp),k, axis=0)[:k]
            row_idx = np.arange(inp.shape[1])
            out[sorted_col_idx, row_idx] = inp[sorted_col_idx, row_idx]
        elif k == np.shape(inp)[0]:
            out = inp
    elif axis == 1:
        if k < np.shape(inp)[1]:
            sorted_row_idx = np.argpartition(-np.abs(inp), k, axis=1)[:k]
            col_idx = np.arange(inp.shape[0])[:, None]
            out[col_idx, sorted_row_idx] = inp[col_idx, sorted_row_idx]
        elif k == np.shape(inp)[1]:
            out = inp
    return out

def keep_k_largest_mags_obsolete(inp, k, axis=0):
    """
    A function to keep only the k elements with the largest magnitudes in an array.

    axis = 0 chooses the largest magnitudes among columns while axis = 1 chooses among rows.
    """
    k = int(k)
    out = np.zeros(np.shape(inp))
    if axis == 0:
        sorted_col_idx = np.argsort(np.abs(inp), axis=0)[inp.shape[0] - k::]
        row_idx = np.arange(inp.shape[1])
        out[sorted_col_idx, row_idx] = inp[sorted_col_idx, row_idx]
    elif axis == 1:
        sorted_row_idx = np.argsort(np.abs(inp), axis=1)[:, inp.shape[1] - k::]
        col_idx = np.arange(inp.shape[0])[:, None]
        out[col_idx, sorted_row_idx] = inp[col_idx, sorted_row_idx]

    return out

#
def nlinearity(Z, param, nlinStrategy):
    """Applying non-linearity to the input Z."""
    if nlinStrategy == 'KBest':
        Phi = keep_k_largest_mags(Z, param)
    elif nlinStrategy == 'KBest_STC':
        Phi = keep_k_largest_mags(Z, param)
        Phi = np.sign(Phi)
    elif nlinStrategy == 'Threshold':
        Phi = (np.abs(Z) >= param) * Z
    elif nlinStrategy == 'Threshold_STC':
        Phi = np.sign((np.abs(Z) >= param) * Z)

    return Phi

def sparsity_pattern_encoder(X,k=None):
    """
    Kind-of run-length encodes the sparsity pattern of STC

    Only for KBest_STC kind of encoding
    """
    X = np.sign(X)
    N = X.shape[1]
    if k is None:
        ks = np.sum(np.abs(X), axis=0)
        assert sum(ks == ks[0]) == N
        k = ks[0]
    # Where are the non-zeros?:
    map_where = np.where(X.T != 0)
    # Their signs:
    map_sign = X[map_where[1], map_where[0]].reshape(k, -1, order='F')
    # Final encoding:
    X_rl = np.diff(
        np.vstack((-np.ones((1, N)),
                   map_where[1].reshape(
                       k, -1, order='F'))), axis=0)*map_sign

    return X_rl.astype('int16')

def sparsity_pattern_decoder(X_rl,n):
    """
    Undoes the sparsity_pattern_encoder
    """
    (k,N) = np.shape(X_rl)
    map_sign = np.sign(X_rl)
    X = np.zeros((n,N))
    X_rl = np.abs(X_rl)
    X_rl[0,:] -= 1
    map_where0 = np.cumsum(X_rl, axis=0).astype(int)
    X[map_where0.reshape(1, -1, order='F'),
      np.repeat(np.arange(N), k)] = map_sign.reshape(1, -1, order='F')
    return X

def qfunction(inp):
    """Apparently numpy doesn't implement the q-function. Here we do it using erfc."""
    out = 0.5 * sps.erfc(np.divide(inp, np.sqrt(2)))
    return out


def opt_ternary_alphabet(Sigma2, threshold):
    """Specifies the optimal non-zero levels for a ternary quantization of a Gaussian RV.

    We assume a zero-mean Gaussian random vector is given whose dimensions are independent
    but with different variances at each dimension. These are specified in Sigma2X vector.
    This vector is going to be scaler quantized with a global threshold to three levels, i.e.,
    {+beta_i, 0,-beta_i} for each dimension. This function calculates beta_i such that the
    distortion of this quantization is minimized. Also the distortion is returned. This answer
    is given in formula (6) of [1].
    """
    dist = lambda beta, sig2, threshold: \
        sig2 + 2 * (beta ** 2 * qfunction(np.divide(threshold, np.sqrt(sig2)))) \
        - 4 * beta * np.sqrt(sig2) * np.divide(np.exp(np.divide(-threshold ** 2, 2 * sig2)), np.sqrt(2 * np.pi))
    # To avoid zero-divisions, we assign small variances as NaN and treat them later accordingly.
    # Any better idea how to avoid these numerical issues?
    Sigma2[Sigma2<=1e-8] = np.nan
    Sigma2[np.divide(threshold,np.sqrt(Sigma2)) >= 1e1] = np.nan

    beta = np.divide(np.sqrt(Sigma2) * np.exp(np.divide(-threshold ** 2, 2 * Sigma2)),
                     np.sqrt(2 * np.pi) * qfunction(np.divide(threshold, np.sqrt(Sigma2))))
    beta[np.isnan(beta)] = 0.
    Distortion = dist(beta, Sigma2, threshold)
    Distortion[np.isnan(Distortion)] = 0
    return beta, Distortion


def ternary_entropy(alpha):
    """Calculates the entropy of an array of random variables with symmetric ternary alphabets.

    For any element of the ternary random vector, alpha_i = Pr[X_i = +beta_i] = Pr[X_i = -beta_i].
    So e.g. Pr[X_i = 0] = 1 - 2*alpha_i. The resulting entropy is calculated for each dimension,
    is measured in bits and is calculated from:
                         H(X) = -2*alpha*log2(alpha) - (1 - 2*alpha)log2(1 - 2*alpha).

    Also note that 0 <= alpha_i <= 0.5"""
    idx_zero = (alpha == 0.0)
    idx_half = (alpha == 0.5)
    alpha[idx_zero] = np.nan
    alpha[idx_half] = np.nan
    #
    H_t = -2 * alpha * np.log2(alpha) - (1 - 2 * alpha) * np.log2(1 - 2 * alpha)
    # For numerical reasons:
    H_t[idx_zero] = 0.
    H_t[idx_half] = 1
    return H_t

def ternary_KLDivergence(alpha_p, alpha_q):
    """
    Calculates the KL-Divergence between two arrays of symmetric ternary random variables.

    For any element of the ternary random vector, alpha_i = Pr[X_i = +beta_i] = Pr[X_i = -beta_i].
    So e.g. Pr[X_i = 0] = 1 - 2*alpha_i. The resulting divergence is calculated for each dimension,
    is measured in bits and is calculated from:
                        D_KL(p||q) =  sum(p(x_i)*log(p(x_i)/q(x_i)))

    Also note that 0 <= alpha_i <= 0.5.

    While the divergence is calculated using the above formula, we should ensure that 0*log2(0) = 2.
    In order to impose this, we consider 5 different scenarios regarding the values of alpha_p
    and alpha_q.
    """
    DKL = lambda alpha_p, alpha_q: 2 * alpha_p * np.log2(np.divide(alpha_p, alpha_q)) + \
                                   (1 - 2 * alpha_p) * np.log2(np.divide((1 - 2 * alpha_p), (1 - 2 * alpha_q)))
    #
    idx_zero_p = (alpha_p == 0.0)
    idx_half_p = (alpha_p == 0.5)
    idx_nz_p = np.logical_not(np.logical_or(idx_zero_p, idx_half_p))
    #
    idx_zero_q = (alpha_q == 0.0)
    idx_half_q = (alpha_q == 0.5)
    idx_nz_q = np.logical_not(np.logical_or(idx_zero_q, idx_half_q))
    # Five different categories out of the nine combinations:
    cat1 = np.logical_and(idx_nz_p, idx_nz_q)
    cat2 = np.logical_or(np.logical_and(idx_zero_p, idx_zero_q),
                         np.logical_and(idx_half_p, idx_half_q))
    cat3 = np.logical_and(idx_zero_p, idx_nz_q)
    cat4 = np.logical_and(idx_half_p, idx_nz_q)
    #
    D = np.inf * np.ones(np.shape(alpha_p))
    D[cat1] = DKL(alpha_p[cat1], alpha_q[cat1])
    D[cat2] = 0.
    D[cat3] = - np.log2(1 - 2 * alpha_q[cat3])
    D[cat4] = -1 - np.log2(alpha_q[cat4])
    #
    return D

def rev_WaterFiller_obsolete(Sigma2,Rate):
    """
    Calculates the optimal "water-lev   el", i.e., the threshold below which no rate is allocated.

    This function is used for rate allocation of independent, but variance-variable Gaussian sources.
    The optimal allocation for such sources requires that no rate should be assigned to sources with
    variances below the water-level and all the sources above this level should be compressed such
    that their distortion reduce to the water-level.
    :param Sigma2: vector of variances
    :param Rate: a range of rates
    :return: thrsh,actv_map,Sigma2_hat,residual_var
    """
    n = Sigma2.size
    # I should check what happens for the iid case.
    thrsh_list = np.linspace(np.maximum(0,0.5*np.min(Sigma2)),np.max(Sigma2),10e4)
    rate_hat = lambda Sigma2, thrsh: np.sum(0.5 * (1/Sigma2.size) *np.log2(np.maximum(1, Sigma2 / thrsh)))
    rate_dev = []
    for ind in thrsh_list:
        rate_dev.append(np.abs(Rate - rate_hat(Sigma2, ind)))
    thrsh = thrsh_list[np.argmin(rate_dev)]
    actv_map = np.where(Sigma2>=thrsh)[0]
    Sigma2_hat = np.zeros_like(Sigma2)
    Sigma2_hat[actv_map] = Sigma2[actv_map] - thrsh
    residual_var = np.copy(Sigma2)
    residual_var[actv_map] = thrsh
    return thrsh,actv_map,Sigma2_hat,residual_var

def rate_estimator_obsolete(Sigma2,nlinParam,nlinStrategy='KBest_STC'):
    '''
    Before doing the reverse-water-filling, we need to estimate the rate of representation.

    This is a very crude approximation though, since we don't know beforehand what
    will be variance profile after the projection.
    :param Sigma2:
    :param nlinParam:
    :param nlinStrategy:
    :return:
    '''
    if nlinStrategy == 'Threshold_STC':
        thrsh = nlinParam
    elif nlinStrategy == 'KBest_STC':
        # First we'll have to estimate an equivalent threshold
        thrsh_list = np.linspace(0,10*np.max(Sigma2),1e4)
        thrsh_dev = []
        for thrsh in thrsh_list:
            thrsh_dev.append(np.abs(nlinParam - 2*np.sum(qfunction(thrsh/np.sqrt(Sigma2[Sigma2>=1e-8])))))
        thrsh = thrsh_list[np.argmin(thrsh_dev)]
    alpha = qfunction(thrsh / np.sqrt(Sigma2[Sigma2>=1e-8]))
    Rate = np.sum(ternary_entropy(alpha)) / Sigma2.size
    return Rate,thrsh


def rev_WaterFiller(Sigma2,Rate):
    """
    Calculates the optimal "water-level", i.e., the threshold below which no rate is allocated.

    This function is used for rate allocation of independent, but variance-variable Gaussian sources.
    The optimal allocation for such sources requires that no rate should be assigned to sources with
    variances below the water-level and all the sources above this level should be compressed such
    that their distortion reduce to the water-level.
    :param Sigma2: vector of variances
    :param Rate: a range of rates
    :return: thrsh,actv_map,Sigma2_hat,residual_var
    """
    #Sigma2 = Sigma2[np.where(Sigma2>=1e-7)]
    rate_dev = lambda thrsh: \
        np.abs(np.sum(0.5 * (1 / Sigma2.size) * np.log2(np.maximum(1, Sigma2 / thrsh))) - Rate)
    # The following lines look like a mess, but quite effective in guaranteeing a
    # decent best solution from a combination of different candidate solutions. This
    # is required due to the numerical issues that pop-up at different rate regimes.
    thrsh_list = []
    thrsh_dev = []
    thrsh_list.append(minimize(rate_dev, 1e-15, method='BFGS', tol=1e-20).x[0])
    thrsh_dev.append(rate_dev(thrsh_list[-1]))
    thrsh_list.append(minimize(rate_dev, 1e-18, method='TNC', tol=1e-22).x[0])
    thrsh_dev.append(rate_dev(thrsh_list[-1]))
    thrsh_list.append(minimize(rate_dev,1e-8,  method='SLSQP',  tol=1e-20).x[0])
    thrsh_dev.append(rate_dev(thrsh_list[-1]))
    thrsh_list.append(minimize(rate_dev,np.min(Sigma2),  method='SLSQP',  tol=1e-20).x[0])
    thrsh_dev.append(rate_dev(thrsh_list[-1]))
    thrsh_list.append(minimize(rate_dev,np.mean(Sigma2),  method='SLSQP',  tol=1e-20).x[0])
    thrsh_dev.append(rate_dev(thrsh_list[-1]))
    thrsh_list.append(minimize(rate_dev,1e-8,  method='SLSQP',  tol=1e-20).x[0])
    thrsh_dev.append(rate_dev(thrsh_list[-1]))
    thrsh_list.append(minimize(rate_dev, np.max(Sigma2), method='SLSQP', tol=1e-20).x[0])
    thrsh_dev.append(rate_dev(thrsh_list[-1]))
    thrsh_list.append(minimize_scalar(rate_dev, bounds=(np.min(Sigma2), np.max(Sigma2)),
                            method='bounded',options={ 'maxiter': 500, 'xatol': 1e-15}).x)
    thrsh_dev.append(rate_dev(thrsh_list[-1]))
    thrsh_dev.append(rate_dev(thrsh_list[-1]))
    thrsh = thrsh_list[np.argmin(thrsh_dev)]
    # thrsh *= 1.5
    actv_map = np.where(Sigma2>=thrsh)[0]
    Sigma2_hat = np.zeros_like(Sigma2)
    Sigma2_hat[actv_map] = Sigma2[actv_map] - thrsh
    residual_var = np.copy(Sigma2)
    residual_var[actv_map] = thrsh
    return thrsh,actv_map,Sigma2_hat,residual_var


def rate_estimator(Sigma2,nlinParam,nlinStrategy='KBest_STC'):
    '''
    Before doing the reverse-water-filling, we need to estimate the rate of representation.

    This is a very crude approximation though, since we don't know beforehand what
    will be variance profile after the projection.
    :param Sigma2:
    :param nlinParam:
    :param nlinStrategy:
    :return:
    '''
    if nlinStrategy == 'Threshold_STC':
        thrsh = nlinParam
    elif nlinStrategy == 'KBest_STC':
        # First we'll have to estimate an equivalent threshold

        dev = lambda thrsh: np.abs(nlinParam - 2 * np.sum(qfunction(thrsh / np.sqrt(Sigma2[Sigma2 >= 1e-8]))))
        # Some cheap heuristic scan btw/ the local minima:
        thrsh_list = []
        thrsh_dev = []
        thrsh_list.append(minimize(dev,0,  method='BFGS',  tol=1e-9).x[0])
        thrsh_dev.append(dev(thrsh_list[-1]))
        thrsh_list.append(minimize(dev, np.mean(Sigma2), method='BFGS', tol=1e-9).x[0])
        thrsh_dev.append(dev(thrsh_list[-1]))
        thrsh_list.append(minimize(dev, np.max(Sigma2), method='BFGS', tol=1e-9).x[0])
        thrsh_dev.append(dev(thrsh_list[-1]))
        thrsh_list.append(minimize(dev, 2*np.max(Sigma2), method='BFGS', tol=1e-9).x[0])
        thrsh_dev.append(dev(thrsh_list[-1]))
        thrsh_list.append(minimize(dev, 10*np.max(Sigma2), method='BFGS', tol=1e-9).x[0])
        thrsh_dev.append(dev(thrsh_list[-1]))
        thrsh_list.append(minimize_scalar(dev, bounds=(1e-8,2*np.max(Sigma2)),
                                method='bounded',options={'disp': 0, 'maxiter': 500, 'xatol': 1e-09}).x)
        thrsh_dev.append(dev(thrsh_list[-1]))

        thrsh = thrsh_list[np.argmin(thrsh_dev)]
    alpha = qfunction(thrsh / np.sqrt(Sigma2[Sigma2>=1e-8]))
    Rate = np.sum(ternary_entropy(alpha)) / Sigma2.size
    return Rate,thrsh

def categorical_entropy(probs):
    """
    Calculates the entropy of a categorical distribution, provided the probability of each item.

    probs is an array of probabilities of each element in the alphabet of length |probs|.
    """
    probs /= np.sum(probs)
    probs = np.clip(probs, 1e-9, 1)
    probs /= np.sum(probs)
    H = - np.sum(probs * np.log2(probs))
    return H


def combinatorial_binomial_poisson_distributions(p_nz, k):
    """
    A binary vector of length n, has n-choose-k combinations with k non-zeros, each with different probability.

    It's a hell of combinatorics!. Beware of n-choose-k!

    p_nz is the probability of ones (non-zeros in general) and its length = n.
    """
    n = len(p_nz)
    combs = list(combinations(np.arange(n), k))
    assert len(combs) <= 1000000
    probs = []
    for i in range(len(combs)):
        tmp_zeros = 1 - p_nz
        tmp_zeros[list(combs[i])] = 1
        zeros_prob = np.prod(tmp_zeros)
        #
        ones_prob = np.prod(p_nz[list(combs[i])])
        #
        probs.append(zeros_prob * ones_prob)
    probs /= np.sum(probs)
    return probs

def SubbandPCA_whitner(F_color,numSB,EigVecs=None,dim_means=None):
    """
    Whitens the correlated input using sub-bands of PCA rotation
    :param F_color: a matrix (numpy array)
    :param numSB: number of sub-bands
    :return: F_white, EigVecs
    """
    if dim_means is None:
        dim_means = np.mean(F_color, axis=1, keepdims=True)
    F_color -= dim_means
    n = F_color.shape[0]
    lenSB = np.floor(n / numSB)
    mapSB = np.append(np.arange(0, lenSB * numSB, lenSB),n)
    mapSB = mapSB.astype(int)
    F_white = np.zeros_like(F_color)
    if EigVecs is not None:
        for sb in range(len(mapSB) - 1):
            F_white[mapSB[sb]:mapSB[sb + 1], :] = EigVecs[sb] @ F_color[mapSB[sb]:mapSB[sb + 1], :]
    else:
        EigVecs = []
        for sb in range(len(mapSB) - 1):
            Cov_mat = np.cov(F_color[mapSB[sb]:mapSB[sb + 1], :])
            Sigma2, U = spla.eigh(Cov_mat)
            idx = Sigma2.argsort()[::-1]
            Sigma2 = Sigma2[idx]
            U = U[:, idx]
            F_white[mapSB[sb]:mapSB[sb + 1], :] = U.T @ F_color[mapSB[sb]:mapSB[sb + 1], :]
            EigVecs.append(U.T)
    return F_white,EigVecs,dim_means

def SubbandPCA_dewhitner(F_white,EigVecs,dim_means):
    """
    Undoes the subband PCA whitening
    :param F_white:
    :param EigVecs:
    :return:
    """
    n = F_white.shape[0]
    numSB = len(EigVecs)
    lenSB = np.ceil(n / numSB)
    mapSB = np.append(np.arange(0, n, lenSB),n)
    mapSB = mapSB.astype(int)
    F_color = np.zeros_like(F_white)
    for sb in range(len(mapSB) - 1):
        F_color[mapSB[sb]:mapSB[sb + 1], :] = EigVecs[sb].T @ F_white[mapSB[sb]:mapSB[sb + 1], :]
    F_color += dim_means
    return F_color


def losers_dump(F_in, ratio, keep_map=None):
    """
    Simply removes small values.

    For super high dimensional data, after whittening according to the current framework,
    it turns out that even for good quality reconstruction, a large portion of the whittened
    data will not be used at all, even after lots of layers. So here we get rid of them from
    the start.
    :param F_in: White data, with super strong decaying profile for variance.
    :param ratio: ratio of cutting point to the largest variance
    :param keep_map: If we already know where we want to cut.
    :return:
    """
    n = F_in.shape[0]
    if keep_map is None:
        Sigma2 = np.var(F_in, axis=1)
        keep_map = np.where(Sigma2 >= ratio * np.max(Sigma2))[0]
    F_out = F_in[keep_map, :]
    norm_sq_res = np.linalg.norm(F_in) ** 2 - np.linalg.norm(F_out) ** 2
    norm_sq_res /= (n - F_out.shape[0]) * F_in.shape[1]
    return F_out, keep_map, n, norm_sq_res


def losers_back(F_in, keep_map, n):
    """
    Undoes the above function by putting zeros at the place of the removed losers.
    :param F_in:
    :param keep_map:
    :param n:
    :return:
    """
    F_out = np.zeros((n, F_in.shape[1]))
    F_out[keep_map, :] = F_in
    return F_out
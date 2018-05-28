import numpy as np
from scipy.spatial.distance import cdist
import scipy.sparse as sps
##################################3
def ternaryVoter(X,Y,nu,nu_p):
    """
    Finds the most similar columns in X for every column of Y with voting.

    This is not optimal at all, not even parallelized. It should be re-written entirely.
    :param X: a sparse matrix with ternary values
    :param Y: a sparse matrix with ternary values
    :param nu: a constant
    :param nu_p: a constant
    :return: Votes
    """
    nu_p = np.abs(nu_p)
    Votes =  (Y > 0).astype(np.int32).T @ (X > 0).astype(np.int32) * nu
    Votes += (Y < 0).astype(np.int32).T @ (X < 0).astype(np.int32) * nu
    #
    Votes -= (Y > 0).astype(np.int32).T @ (X < 0).astype(np.int32) * nu_p
    Votes -= (Y < 0).astype(np.int32).T @ (X > 0).astype(np.int32) * nu_p


    return Votes.T
########################
def exact_distance_matcher_inefficient(database, query):
    """
    Non-vectorized version, very inefficient, but to be used for list refinement.
    """
    from scipy.spatial.distance import cdist
    distances = cdist(query.reshape(1, -1), database.T)
    list_refined = np.argsort(distances, axis=1).T
    return list_refined


###################
def List_refiner(F_hat, Q, list_initial):
    """
    I'm sorry, I will have to use a for-loop here!
    """
    LIST_REFINED = np.array([]).reshape(np.shape(list_initial)[0], 0)
    for i in range(np.shape(Q)[1]):
        database = F_hat[:, list_initial[:, i]]
        list_refined = exact_distance_matcher_inefficient(database, Q[:, i])
        list_refined = list_initial[list_refined, i]
        LIST_REFINED = np.hstack((LIST_REFINED, list_refined.reshape(-1, 1)))
    return LIST_REFINED
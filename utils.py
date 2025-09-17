import os
import sys
import numpy as np
import healpy as hp
#import healsparse
import scipy as scipy
from scipy.spatial import SphericalVoronoi
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm


def get_bins(lower, upper, nbins, return_edges = False):

    binEdges = np.linspace(lower, upper, nbins + 1 ) 
    binMid = ( binEdges[1:] + binEdges[:-1] ) / 2.

    if return_edges:
        return binMid, binEdges
    else:
        return binMid
    
    
    
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]



def fill_nan(input_y):
    y = np.copy(input_y)
    for i in range(y.shape[0]):
        nans, x= nan_helper(y[i])
        y[i,nans] = np.interp(x(nans), x(~nans), y[i,~nans])
    return y

def nansumwrapper(a, **kwargs):
    if np.isnan(a).all():
        return np.nan
    else:
        return np.nansum(a, **kwargs)
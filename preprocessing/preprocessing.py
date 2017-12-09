from sklearn.preprocessing import scale, normalize, StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Scale data
def scale(data):
    return scale(data)

# Normalize data
def st_scale(data):
    return StandardScaler(with_mean=True).fit_transform(data)

# Normalize data
def normalize_l2(data):
    return normalize(data, norm = 'l2')

def PCA_reduction(data, var, ncomp = None):
    # If number of components not expecified use variance to find
    if ncomp == None:
        # Use PCA with all components to get variance values
        pca = PCA(n_components = 2209)
        pca.fit_transform(data)
        variances = pca.explained_variance_ratio_

        # Suming values for number of components
        for i in range(1, len(variances)):
            variances[i] = variances[i] + variances[i-1]

        # Get first component higher than variance expected
        comp = np.where(variances >= var)[0][0] + 1
    # Using number of components expecified on parameter
    else:
        comp = ncomp

    # Executing PCA with number of components found
    pca = PCA(n_components = comp)
    reduction = pca.fit_transform(data)

    return reduction, comp

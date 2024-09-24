import numpy as np
from numpy import errstate, true_divide, isfinite, isscalar
from scipy import sparse


def split_file(mat, length, lap):
    mat2 = np.zeros(shape=(length, length))
    mat3 = []
    for i in range(0, len(mat) - length, lap):
        t = i
        for k in range(t, min(t + length, len(mat))):
            for j in range(t, min(t + length, len(mat))):
                mat2[k - i][j - i] = mat[k][j]
        mat3.append(list(mat2))
        mat2 = np.zeros(shape=(length, length))
    for j in mat3:
        for i in range(0, len(j) - 1):
            j[i][i] = 0
            j[i][i + 1] = 0
            j[i + 1][i] = 0
        j[len(j) - 1][len(j) - 1] = 0
    return mat3


def transform_matrix(two_dim, filter_p=0.5):
    tri_list = []
    for i in range(len(two_dim)):
        for j in range(len(two_dim)):
            if two_dim[i][j] != 0:
                if two_dim[i][j] > filter_p:
                    tri_list.append([i + 1, j + 1, filter_p])
                else:
                    tri_list.append([i + 1, j + 1, two_dim[i][j]])
    return tri_list


def read_Infomap_result(file, window_step, min_bin_len=3):
    with open(file, 'r') as f:
        bounds_module = []
        for line in f.readlines():
            newline = line.rstrip('\n').split(' ')
            if newline[1] == "partitioned":
                module_num = int(newline[6])
                for i in range(module_num):
                    bounds_module.append([])
            if newline[0] != '#' and newline[0][1] == ':':
                bounds_module[int(newline[0][0]) - 1].append(int(newline[-1]) + window_step)
    for i in range(module_num):
        bounds_module[i] = sorted(bounds_module[i])
    bounds_module = sorted(bounds_module)

    modules_new = []
    for module in bounds_module:
        diff = np.array(module[1:]) - np.array(module[:-1])
        non_one_indices = np.where((diff != 1) & (diff != 2))[0]
        if len(non_one_indices) == 0:
            modules_new.append(module)
        elif len(non_one_indices) == 1:
            module_left = module[:non_one_indices[0] + 1]
            module_right = module[non_one_indices[0] + 1:]
            if len(module_left) >= min_bin_len:
                modules_new.append(module_left)
            if len(module_right) >= min_bin_len:
                modules_new.append(module_right)
        else:
            for i in range(len(non_one_indices)):
                if i == 0:
                    module_left = module[:non_one_indices[i] + 1]
                    module_right = module[non_one_indices[i] + 1:non_one_indices[i + 1]]
                    if len(module_left) >= min_bin_len:
                        modules_new.append(module_left)
                    if len(module_right) >= min_bin_len:
                        modules_new.append(module_right)
                elif 0 < i < len(non_one_indices) - 1:
                    module_middle = module[non_one_indices[i] + 1:non_one_indices[i + 1] + 1]
                    if len(module_middle) >= min_bin_len:
                        modules_new.append(module_middle)
                else:
                    module_end = module[non_one_indices[i] + 1:]
                    if len(module_end) >= min_bin_len:
                        modules_new.append(module_end)
    bounds_module_new = []
    for module in modules_new:
        if len(module) >= min_bin_len:
            continuity_rate = len(module) / (module[-1] - module[0] + 1)
            if continuity_rate > 0.75:
                bounds_module_new.append(module)

    if bounds_module_new:
        bounds = [(bounds_module_new[0][0] - 1, bounds_module_new[0][0]),
                  (bounds_module_new[-1][-1], bounds_module_new[-1][-1] + 1)]
        for i in range(len(bounds_module_new) - 1):
            left = bounds_module_new[i][-1]
            right = bounds_module_new[i + 1][0]
            if left - right == 1:
                bounds.append((right, left))
                bounds_module_new[i][-1] = right
                bounds_module_new[i + 1][0] = left
            elif right - left == 1:
                bounds.append((left, right))
            else:
                bounds.append((left, left + 1))
                bounds.append((right - 1, right))
        bounds = sorted(bounds, key=lambda x: x[0])
        return bounds
    else:
        return []


def read_Infomap_result_start(file, window_step, min_bin_len=3):
    with open(file, 'r') as f:
        bounds_module = []
        for line in f.readlines():
            newline = line.rstrip('\n').split(' ')
            if newline[1] == "partitioned":
                module_num = int(newline[6])
                for i in range(module_num):
                    bounds_module.append([])
            if newline[0] != '#' and newline[0][1] == ':':
                bounds_module[int(newline[0][0]) - 1].append(int(newline[-1]) + window_step)
    for i in range(module_num):
        bounds_module[i] = sorted(bounds_module[i])
    bounds_module = sorted(bounds_module)
    modules_new = []
    for module in bounds_module:
        diff = np.array(module[1:]) - np.array(module[:-1])
        non_one_indices = np.where((diff != 1) & (diff != 2))[0]
        if len(non_one_indices) == 0:
            modules_new.append(module)
        elif len(non_one_indices) == 1:
            module_left = module[:non_one_indices[0] + 1]
            module_right = module[non_one_indices[0] + 1:]
            if len(module_left) >= min_bin_len:
                modules_new.append(module_left)
            if len(module_right) >= min_bin_len:
                modules_new.append(module_right)
        else:
            for i in range(len(non_one_indices)):
                if i == 0:
                    module_left = module[:non_one_indices[i] + 1]
                    module_right = module[non_one_indices[i] + 1:non_one_indices[i + 1]]
                    if len(module_left) >= min_bin_len:
                        modules_new.append(module_left)
                    if len(module_right) >= min_bin_len:
                        modules_new.append(module_right)
                elif 0 < i < len(non_one_indices) - 1:
                    module_middle = module[non_one_indices[i] + 1:non_one_indices[i + 1] + 1]
                    if len(module_middle) >= min_bin_len:
                        modules_new.append(module_middle)
                else:
                    module_end = module[non_one_indices[i] + 1:]
                    if len(module_end) >= min_bin_len:
                        modules_new.append(module_end)
    bounds_module_new = []
    for module in modules_new:
        if len(module) >= min_bin_len:
            continuity_rate = len(module) / (module[-1] - module[0] + 1)
            if continuity_rate > 0.75:
                bounds_module_new.append(module)
    if bounds_module_new:
        bounds = [(bounds_module_new[0][0] - 1, bounds_module_new[0][0]),
                  (bounds_module_new[-1][-1], bounds_module_new[-1][-1] + 1)]
        for i in range(len(bounds_module_new) - 1):
            left = bounds_module_new[i][-1]
            right = bounds_module_new[i + 1][0]
            if left - right == 1:
                bounds.append((right, left))
                bounds_module_new[i][-1] = right
                bounds_module_new[i + 1][0] = left
            elif right - left == 1:
                bounds.append((left, right))
            else:
                bounds.append((left, left + 1))
                bounds.append((right - 1, right))
        bounds = sorted(bounds, key=lambda x: x[0])
        return bounds_module_new[0][0], bounds
    else:
        if modules_new:
            return modules_new[0][0], []
        else:
            return bounds_module[0][0], []


def read_Infomap_result_end(file, window_step, min_bin_len=3):
    with open(file, 'r') as f:
        bounds_module = []
        for line in f.readlines():
            newline = line.rstrip('\n').split(' ')
            if newline[1] == "partitioned":
                module_num = int(newline[6])
                for i in range(module_num):
                    bounds_module.append([])
            if newline[0] != '#' and newline[0][1] == ':':
                bounds_module[int(newline[0][0]) - 1].append(int(newline[-1]) + window_step)
    for i in range(module_num):
        bounds_module[i] = sorted(bounds_module[i])
    bounds_module = sorted(bounds_module)
    modules_new = []
    for module in bounds_module:
        diff = np.array(module[1:]) - np.array(module[:-1])
        non_one_indices = np.where((diff != 1) & (diff != 2))[0]
        if len(non_one_indices) == 0:
            modules_new.append(module)
        elif len(non_one_indices) == 1:
            module_left = module[:non_one_indices[0] + 1]
            module_right = module[non_one_indices[0] + 1:]
            if len(module_left) >= min_bin_len:
                modules_new.append(module_left)
            if len(module_right) >= min_bin_len:
                modules_new.append(module_right)
        else:
            for i in range(len(non_one_indices)):
                if i == 0:
                    module_left = module[:non_one_indices[i] + 1]
                    module_right = module[non_one_indices[i] + 1:non_one_indices[i + 1]]
                    if len(module_left) >= min_bin_len:
                        modules_new.append(module_left)
                    if len(module_right) >= min_bin_len:
                        modules_new.append(module_right)
                elif 0 < i < len(non_one_indices) - 1:
                    module_middle = module[non_one_indices[i] + 1:non_one_indices[i + 1] + 1]
                    if len(module_middle) >= min_bin_len:
                        modules_new.append(module_middle)
                else:
                    module_end = module[non_one_indices[i] + 1:]
                    if len(module_end) >= min_bin_len:
                        modules_new.append(module_end)
    bounds_module_new = []
    for module in modules_new:
        if len(module) >= min_bin_len:
            continuity_rate = len(module) / (module[-1] - module[0] + 1)
            if continuity_rate > 0.75:
                bounds_module_new.append(module)
    if bounds_module_new:
        bounds = [(bounds_module_new[0][0] - 1, bounds_module_new[0][0]),
                  (bounds_module_new[-1][-1], bounds_module_new[-1][-1] + 1)]
        for i in range(len(bounds_module_new) - 1):
            left = bounds_module_new[i][-1]
            right = bounds_module_new[i + 1][0]
            if left - right == 1:
                bounds.append((right, left))
                bounds_module_new[i][-1] = right
                bounds_module_new[i + 1][0] = left
            elif right - left == 1:
                bounds.append((left, right))
            else:
                bounds.append((left, left + 1))
                bounds.append((right - 1, right))
        bounds = sorted(bounds, key=lambda x: x[0])
        return bounds_module_new[-1][-1], bounds
    else:
        if modules_new:
            return modules_new[-1][-1], []
        else:
            return bounds_module[-1][-1], []


def merge_TAD(bounds_dict, bounds_right):
    if len(bounds_dict) == 0:
        bounds = dict()
        for bound in bounds_right:
            bounds[(bound[0], bound[1])] = 1
        return bounds
    else:
        for bounds in bounds_right:
            if bounds not in bounds_dict:
                bounds_dict[(bounds[0], bounds[1])] = 1
            else:
                bounds_dict[bounds] += 1
        return bounds_dict


def SCN(M, **kwargs):
    """ Performs Sequential Component Normalization on matrix *M*.

    .. [AC12] Cournac A, Marie-Nelly H, Marbouty M, Koszul R, Mozziconacci J.
       Normalization of a chromosomal contact map. *BMC Genomics* **2012**.
    """

    total_count = kwargs.pop('total_count', None)
    max_loops = kwargs.pop('max_loops', 100)
    tol = kwargs.pop('tol', 1e-5)

    N = M.copy()
    n = 0
    d0 = None
    p = 1
    last_p = None

    while True:
        C = np.diag(div0(1., np.sum(N, axis=0)))
        N = np.dot(N, C)

        R = np.diag(div0(1., np.sum(N, axis=1)))
        N = np.dot(R, N)

        n += 1

        # check convergence of symmetry
        d = np.mean(np.abs(N - N.T))

        if d0 is not None:
            p = div0(d, d0)
            dp = np.abs(p - last_p)
            if dp < tol:
                break
        else:
            d0 = d
        last_p = p

        if max_loops is not None:
            if n >= max_loops:
                break
    # guarantee symmetry
    N = (N + N.T) / 2.
    if total_count == 'original':
        total_count = np.sum(M)

    if total_count is not None:
        sum_N = np.sum(N)
        k = total_count / sum_N
        N = N * k
    return N


def div0(a, b, defval=0.):
    with errstate(divide='ignore', invalid='ignore'):
        c = true_divide(a, b)
        if isscalar(c):
            if not isfinite(c):
                c = defval
        else:
            c[~isfinite(c)] = defval
    return c


def KR(matrix, tol=1e-12, x0=None, delta=0.1, Delta=3, fl=0):
    ''' Knight-Ruiz algorithm for matrix balancing
    This code is from Rajendra,K. et al. (2017) Genome contact map explorer: a platform for the comparison,
    interactive visualization and analysis of genome contact maps. Nucleic Acids Research, 45(17):e152.
    '''

    bNoData = np.all(matrix == 0.0, axis=0)
    bNonZeros = ~bNoData
    A = (matrix[bNonZeros, :])[:, bNonZeros]  # Selected row-column which are not all zeros
    # perform KR methods
    n = A.shape[0]  # n = size(A,1)
    e = np.ones((n, 1))  # e = ones(n,1)
    res = []
    if x0 is None:
        x0 = e

    g = 0.9  # Parameters used in inner stopping criterion.
    etamax = 0.1  # Parameters used in inner stopping criterion.
    eta = etamax
    stop_tol = tol * 0.5
    x = x0
    rt = tol ** 2  # rt = tol^2
    v = x * A.dot(x)  # v = x.*(A*x)
    rk = 1 - v
    rho_km1 = np.dot(rk.conjugate().T, rk)  # rho_km1 = rk'*rk
    rout = rho_km1
    rold = rout

    # x, x0, e, v, rk, y, Z, w, p, ap :     vector shape(n, 1) : [ [value] [value] [value] [value] ... ... ... [value] ]
    # rho_km1, rout, rold, innertol, alpha :  scalar shape(1 ,1) : [[value]]

    MVP = 0  # count matrix vector products.
    i = 0  # Outer iteration count.

    if fl == 1:
        print('it in. it res')

    while rout > rt:  # Outer iteration
        i = i + 1
        k = 0
        y = e
        innertol = max([eta ** 2 * rout, rt])  # innertol = max([eta^2*rout,rt]);

        while rho_km1 > innertol:  # Inner iteration by CG
            k = k + 1
            if k == 1:
                Z = rk / v  # Z = rk./v
                p = Z
                rho_km1 = np.dot(rk.conjugate().T, Z)  # rho_km1 = rk'*Z
            else:
                beta = rho_km1 / rho_km2
                p = Z + (beta * p)

            # Update search direction efficiently.
            w = x * A.dot((x * p)) + (v * p)  # w = x.*(A*(x.*p)) + v.*p
            alpha = rho_km1 / np.dot(p.conjugate().T, w)  # alpha = rho_km1/(p'*w)
            ap = alpha * p  # ap = alpha*p (No dot function as alpha is scalar)

            # Test distance to boundary of cone.
            ynew = y + ap;
            # print(i, np.amin(ynew), delta, np.amin(ynew) <= delta)
            # print(i, np.amax(ynew), Delta, np.amax(ynew) >= Delta)
            if np.amin(ynew) <= delta:
                if delta == 0:
                    break
                ind = np.nonzero(ap < 0)  # ind = find(ap < 0)
                gamma = np.amin((delta - y[ind]) / ap[ind])  # gamma = min((delta - y(ind))./ap(ind))
                y = y + np.dot(gamma, ap)  # y = y + gamma*ap
                break
            if np.amax(ynew) >= Delta:
                ind = np.nonzero(ynew > Delta)  # ind = find(ynew > Delta);
                gamma = np.amin((Delta - y[ind]) / ap[ind])  # gamma = min((Delta-y(ind))./ap(ind));
                y = y + np.dot(gamma, ap)  # y = y + gamma*ap;
                break
            y = ynew
            rk = rk - alpha * w  # rk = rk - alpha*w
            rho_km2 = rho_km1
            Z = rk / v
            rho_km1 = np.dot(rk.conjugate().T, Z)  # rho_km1 = rk'*Z

        x = x * y  # x = x.*y
        v = x * A.dot(x)  # v = x.*(A*x)
        rk = 1 - v
        rho_km1 = np.dot(rk.conjugate().T, rk)  # rho_km1 = rk'*rk
        rout = rho_km1
        MVP = MVP + k + 1

        # Update inner iteration stopping criterion.
        rat = rout / rold
        rold = rout
        res_norm = np.sqrt(rout)
        eta_o = eta
        eta = g * rat

        # print(i, res_norm)

        if g * eta_o ** 2 > 0.1:
            eta = np.amax([eta, g * eta_o ** 2])  # eta = max([eta,g*eta_o^2])

        eta = np.amax(
            [np.amin([eta, etamax]), stop_tol / res_norm]);  # eta = max([min([eta,etamax]),stop_tol/res_norm]);

        if fl == 1:
            print('%3d %6d %.3e %.3e %.3e \n' % (i, k, res_norm, np.amin(y), np.amin(x)))
            res = [res, res_norm]

    # Generation of Doubly stochastic matrix ( diag(X)*A*diag(X) )
    outA = x.T * (A * x)

    normMat = np.zeros(matrix.shape)  # all 0 init matrix

    # Store normalized values to output matrix
    dsm_i = 0
    ox = normMat.shape[0]
    idx_fill = np.nonzero(~bNoData)
    for i in range(ox):
        if not bNoData[i]:
            normMat[i, idx_fill] = outA[dsm_i]
            normMat[idx_fill, i] = outA[dsm_i]
            dsm_i += 1

    # Get the maximum and minimum value except zero
    ma = np.ma.masked_equal(outA, 0.0, copy=False)
    n_min = ma.min()
    n_max = ma.max()

    return normMat, n_min, n_max


def SQRTVCnorm(M, **kwargs):
    """ Performs square-root vanilla coverage normalization on matrix *M*."""

    total_count = kwargs.get('total_count', 'original')

    C = np.diag(np.sqrt(div0(1., np.sum(M, axis=0))))
    R = np.diag(np.sqrt(div0(1., np.sum(M, axis=1))))

    # N = R * M * C
    N = np.dot(np.dot(R,M),C)

    if total_count == 'original':
        total_count = np.sum(M)

    if total_count is not None:
        sum_N = np.sum(N)
        k = total_count / sum_N
        N = N * k
    return N


def is_symetric_or_tri(X, eps=1e-7):
    m, n = X.shape
    if m != n:
        raise ValueError("The matrix should be of shape (n, n)")

    if is_tri(X):
        return True
    if np.abs(X - X.T).sum() > eps:
        raise ValueError("The matrix should be symmetric")


def is_tri(X):
    diag = X.diagonal().sum()
    if sparse.issparse(X):
        if not (sparse.tril(X).sum() - diag) or \
           not (sparse.triu(X).sum() - diag):
            return True
    elif not np.triu(X, 1).sum() or not np.tril(X, -1).sum():
        return True
    else:
        return False


def ICE(X, SS=None, max_iter=3000, eps=1e-4, copy=True,
                      norm='l1', verbose=0, output_bias=False,
                      total_counts=None, counts_profile=None):
    """
    ICE normalization

    The imakaev normalization of Hi-C data consists of iteratively estimating
    the bias such that all the rows and columns (ie loci) have equal
    visibility.

    Parameters
    ----------
    X : ndarray or sparse array (n, n)
        raw interaction frequency matrix

    max_iter : integer, optional, default: 3000
        Maximum number of iteration

    eps : float, optional, default: 1e-4
        the relative increment in the results before declaring convergence.

    copy : boolean, optional, default: True
        If copy is True, the original data is not modified.

    norm : string, optional, default: l1
        If set to "l1", will compute the ICE algorithm of the paper. Else, the
        algorithm is adapted to use the l2 norm, as suggested in the SCN
        paper.

    output_bias : boolean, optional, default: False
        whether to output the bias vector.

    total_counts : float, optional, default: None
        the total number of contact counts that the normalized matrix should
        contain. If set to None, the normalized contact count matrix will be
        such that the total number of contact counts equals the initial number
        of interactions.

    Returns
    -------
    X, (bias) : ndarray (n, n)
        Normalized IF matrix and bias of output_bias is True

    Example
    -------
    .. plot:: examples/normalization/plot_ICE_normalization.py
    """
    if copy:
        X = X.copy()

    if sparse.issparse(X):
        if not sparse.isspmatrix_coo(X):
            X = sparse.coo_matrix(X, dtype=float)
    else:
        X[np.isnan(X)] = 0
    X = X.astype('float')

    m = X.shape[0]
    is_symetric_or_tri(X)
    old_bias = None
    bias = np.ones((m, 1))
    _is_tri = is_tri(X)
    if verbose:
        print("Matrix is triangular superior")

    if counts_profile is not None:
        rows_to_remove = counts_profile == 0
        if sparse.issparse(X):
            rows_to_remove = np.where(rows_to_remove)[0]
            X.data[np.isin(X.row, rows_to_remove)] = 0
            X.data[np.isin(X.col, rows_to_remove)] = 0
            X.eliminate_zeros()
        else:
            X[rows_to_remove] = 0
            X[:, rows_to_remove] = 0

    if total_counts is None:
        total_counts = X.sum()
    for it in np.arange(max_iter):
        if norm == 'l1':
            # Actually, this should be done if the matrix is diag sup or diag
            # inf
            if _is_tri:
                sum_ds = X.sum(axis=0) + X.sum(axis=1).T - X.diagonal()
            else:
                sum_ds = X.sum(axis=0)
        elif norm == 'l2':
            if _is_tri:
                sum_ds = ((X**2).sum(axis=0) +
                          (X**2).sum(axis=1).T -
                          (X**2).diagonal())
            else:
                sum_ds = (X**2).sum(axis=0)

        if SS is not None:
            raise NotImplementedError
        dbias = sum_ds.reshape((m, 1))
        if counts_profile is not None:
            dbias /= counts_profile[:, np.newaxis]
            dbias[counts_profile == 0] = 0
        # To avoid numerical instabilities
        dbias /= dbias[dbias != 0].mean()

        dbias[dbias == 0] = 1
        bias *= dbias

        if sparse.issparse(X):
            X.data /= dbias.A[X.row, 0]
            X.data /= dbias.A[X.col, 0]
        else:
            X /= dbias
            X /= dbias.T

        bias *= np.sqrt(X.sum() / total_counts)
        X *= total_counts / X.sum()

        if old_bias is not None and np.abs(old_bias - bias).sum() < eps:
            if verbose > 1:
                print("break at iteration %d" % (it,))
            break

        if verbose > 1 and old_bias is not None:
            print('ICE at iteration %d %s' %
                  (it, np.abs(old_bias - bias).sum()))
        old_bias = bias.copy()
    # Now that we are finished with the bias estimation, set all biases
    # corresponding to filtered rows to np.nan
    if sparse.issparse(X):
        to_rm = (np.array(X.sum(axis=0)).flatten() +
                 np.array(X.sum(axis=1)).flatten()) == 0
    else:
        to_rm = (X.sum(axis=0) + X.sum(axis=1)) == 0
    bias[to_rm] = np.nan
    if output_bias:
        return X, bias
    else:
        return X





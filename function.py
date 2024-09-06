import numpy as np
from numpy import errstate, true_divide, isfinite, isscalar


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

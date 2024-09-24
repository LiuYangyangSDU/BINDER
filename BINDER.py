import sys
import function
import os
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import ranksums
import argparse
import shutil
import warnings
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", message="loadtxt: input contained no data")
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)



class BoundaryDataSet(Dataset):
    def __init__(self, file_name):
        try:
            data = np.loadtxt(file_name, dtype=np.str_, encoding='utf-8')
            if data.size == 0:
                self.feature = torch.tensor([])
                self.label = torch.tensor([])
                self.n_samples = 0
            else:
                data = data[:, :].astype(np.float32)
                self.feature = torch.tensor(data[:, :])
                self.label = torch.tensor(data[:, [data.shape[1] - 1]])
                self.n_samples = data.shape[0]
        except Exception as e:
            print(f"Error reading the file: {e}")
            self.feature = torch.tensor([])
            self.label = torch.tensor([])
            self.n_samples = 0

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.n_samples


class NeuralModel(nn.Module):
    def __init__(self):
        super(NeuralModel, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Linear(220, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.branch1(x[:])
        return out


def getStartAndEndIndex(tri_list):
    start_index, end_index = 0, -1
    for k in range(len(tri_list)):
        if len(tri_list[k]) > 2:
            start_index = k
            break
    tri_list_reverse = tri_list[::-1]
    for j in range(len(tri_list)):
        if len(tri_list_reverse[j]) > 2:
            end_index = len(tri_list) - j - 1
            break
    return start_index, end_index


def get_Infomap_bounds(matrix, map_path, length=55, lap=10):
    mat_list = function.split_file(matrix, length=length, lap=lap)
    tri_list = []
    for mat in mat_list:
        tri_list.append(function.transform_matrix(mat))
    start_index, end_index = getStartAndEndIndex(tri_list=tri_list)
    window_pos = 0
    bounds = dict()
    for i in range(len(tri_list)):
        if len(tri_list[i]) <= 2:
            window_pos += lap
            continue
        with open(map_path + str(window_pos) + ".txt", "w") as fw:
            for ele in tri_list[i]:
                for ele2 in ele:
                    fw.write(str(ele2))
                    fw.write("\t")
                fw.write("\n")
            fw.close()
            if i != start_index and i != end_index:
                subprocess.call(
                    [sys.path[0] +"/Infomap", map_path + str(window_pos) + ".txt", map_path[:-1], "-N", "20", "--silent"])
                TADbdlist = function.read_Infomap_result(map_path + str(window_pos) + ".tree", window_pos)
                if TADbdlist is not None:
                    bounds = function.merge_TAD(bounds, TADbdlist)
            if i == start_index:
                subprocess.call(
                    [sys.path[0] +"/Infomap", map_path + str(window_pos) + ".txt", map_path[:-1], "-N", "20", "--silent"])
                start_bin, TADbdlist = function.read_Infomap_result_start(map_path + str(window_pos) + ".tree", window_pos)
                if TADbdlist is not None:
                    bounds = function.merge_TAD(bounds, TADbdlist)
            if i == end_index:
                subprocess.call(
                    [sys.path[0] +"/Infomap", map_path + str(window_pos) + ".txt", map_path[:-1], "-N", "20", "--silent"])
                end_bin, TADbdlist = function.read_Infomap_result_end(map_path + str(window_pos) + ".tree", window_pos)
                if TADbdlist is not None:
                    bounds = function.merge_TAD(bounds, TADbdlist)
            window_pos += lap

    bounds_sorted = dict(sorted(bounds.items(), key=lambda x: x[0][0]))
    bad_bounds = []
    for bound in bounds_sorted.keys():
        if bound[1] - bound[0] != 1:
            bad_bounds.append(bound)
    for bound in bad_bounds:
        bounds_sorted.pop(bound)
    bound_with_info = dict()
    for bound in bounds_sorted.keys():
        bound_with_info[(bound[0], bound[1])] = bounds_sorted[bound]
    for i in range(start_bin, end_bin):
        dual_bound = (i, i + 1)
        if dual_bound not in bound_with_info.keys():
            bound_with_info[dual_bound] = 0
    return start_bin, end_bin, bound_with_info


def getCTCF(file):
    CTCF_all = dict()
    with open(file, 'r') as f:
        for line in f.readlines():
            newLine = line.rstrip('\n').split('\t')
            if newLine[0] in CTCF_all.keys():
                CTCF_all[newLine[0]].append([int(newLine[1]), int(newLine[2])])
            else:
                CTCF_all[newLine[0]] = [[int(newLine[1]), int(newLine[2])]]
    return CTCF_all


def getBoundToBool(CTCF_file, matrix, resolution, chromatin):
    ctcf_all = getCTCF(CTCF_file)
    boundToBool = dict()
    for b in range(1, len(matrix)):
        interval = [(b - 0.5) * resolution, (b + 0.5) * resolution]
        bound_ctcf = [ctcf for ctcf in ctcf_all[chromatin] if interval[0] <= ctcf[1] and interval[1] >= ctcf[0]]
        if len(bound_ctcf) > 0:
            boundToBool[(b - 1, b)] = 1
        else:
            boundToBool[(b - 1, b)] = 0
    return boundToBool


def getBoundWithInfo(file, start, end):
    bound_with_info = dict()
    raw = np.loadtxt(file).astype(np.int32)
    for i in range(raw.shape[0]):
        bound_with_info[(raw[i][0], raw[i][1])] = raw[i][2]
    for i in range(start, end):
        dual_bound = (i, i + 1)
        if dual_bound not in bound_with_info.keys():
            bound_with_info[dual_bound] = 0
    return bound_with_info


def getWithinRegionForBin(matrix, b, P_window_size):
    return matrix[max(b - P_window_size, 0):b, b:min(b + P_window_size, len(matrix))].flatten()


def getBetweenRegionUpstreamForBin(matrix, b, P_window_size):
    between_region_upstream = np.triu(matrix[max(b - P_window_size, 0):b, max(b - P_window_size, 0):b], k=1)
    indices = np.triu_indices(between_region_upstream.shape[0], k=1)
    return between_region_upstream[indices]


def getBetweenRegionDownstreamForBin(matrix, b, P_window_size):
    between_region_downstream = np.triu(matrix[b:min(b + P_window_size, len(matrix)), b:min(b + P_window_size, len(matrix))], k=1)
    indices = np.triu_indices(between_region_downstream.shape[0], k=1)
    return between_region_downstream[indices]


def getTwoRegionsForBin(matrix, b, P_window_size):
    between_region_upstream = getBetweenRegionUpstreamForBin(matrix, b, P_window_size)
    between_region_downstream = getBetweenRegionDownstreamForBin(matrix, b, P_window_size)
    between_region = np.concatenate((between_region_upstream, between_region_downstream))
    within_region = getWithinRegionForBin(matrix, b, P_window_size)
    return between_region, within_region


def getSignalForBound(b, matrix, window_size):
    return np.mean(matrix[max(b - window_size, 0):b, b:min(b + window_size, len(matrix))])


def getDIForBound(b, matrix, DI_window_size):
    A = np.sum(matrix[b - DI_window_size:b, b+1])
    B = np.sum(matrix[b - 1, b+1:b + DI_window_size + 1])
    E = (A + B) / 2
    DI = (-1) * ((B - A) / abs(B - A)) * (((A - E) ** 2) / E + ((B - E) ** 2) / E)
    return DI


def getSignalFeatureForBound(b, matrix, window_size, binSignal_size):
    binSignal_bin = [0] * 11
    out_index = []
    range_list = range(b - binSignal_size, b + binSignal_size + 1)
    for i in range_list:
        if i <= 0 or i >= len(matrix):
            out_index.append(range_list.index(i))
    for i in range_list:
        if 0 < i < len(matrix):
            binSignal_bin[range_list.index(i)] = getSignalForBound(i, matrix, window_size)
    if len(out_index) != 0:
        if out_index[0] <= 0:
            for index in out_index:
                binSignal_bin[index] = binSignal_bin[out_index[-1] + 1]
        else:
            for index in out_index:
                binSignal_bin[index] = binSignal_bin[out_index[0] - 1]
    return binSignal_bin


def getDIFeatureForBound(b, matrix, DI_window_size, DI_size):
    DI_feature_for_bin = [0] * 10
    out_index = []
    range_list = range(b - DI_size, b + DI_size)
    for i in range_list:
        if i <= 0 or i >= len(matrix) - 1:
            out_index.append(range_list.index(i))
    for i in range_list:
        if 0 < i < len(matrix) - 1:
            DI_feature_for_bin[range_list.index(i)] = getDIForBound(i, matrix, DI_window_size)
    for i in range(len(DI_feature_for_bin)):
        if np.isnan(DI_feature_for_bin[i]):
            DI_feature_for_bin[i] = 0
    if len(out_index) != 0:
        if out_index[0] <= 0:
            for index in out_index:
                DI_feature_for_bin[index] = DI_feature_for_bin[out_index[-1] + 1]
        else:
            for index in out_index:
                DI_feature_for_bin[index] = DI_feature_for_bin[out_index[0] - 1]
    return DI_feature_for_bin


def getSignalFeature(bound_list, matrix, window_size, binSignal_size):
    boundToSignalFeature = dict()
    for bound in bound_list:
        b = bound[1]
        boundToSignalFeature[(b-1, b)] = getSignalFeatureForBound(b, matrix, window_size, binSignal_size)
    return boundToSignalFeature


def getDIFeature(bound_list, matrix, DI_window_size, DI_size):
    boundToDIFeature = dict()
    for bound in bound_list:
        b = bound[1]
        boundToDIFeature[(b-1, b)] = getDIFeatureForBound(b, matrix, DI_window_size, DI_size)
    return boundToDIFeature


def getPvalueForBin(matrix, b, P_window_size):
    between_region, within_region = getTwoRegionsForBin(matrix, b, P_window_size)
    _, p_value = ranksums(between_region, within_region)
    difference = np.mean(between_region) - np.mean(within_region)
    return p_value


def getPvalueFeature(bound_list, matrix, P_window_size):
    PvalueFeature = dict()
    for bound in bound_list:
        b = bound[1]
        PvalueFeature[(b - 1, b)] = [getPvalueForBin(matrix, b, P_window_size)]
    return PvalueFeature


def getFivePartBounds(bound_with_info, min_info_support):
    bound_list_1, bound_list_2, bound_list_3, bound_list_4, bound_list_5 = [], [], [], [], []
    for key, value in bound_with_info.items():
        if value == 0:
            bound_list_1.append(key)
        if value == 1:
            bound_list_2.append(key)
        if value == 2:
            bound_list_3.append(key)
        if 2 < value < min_info_support:
            bound_list_4.append(key)
        if value >= min_info_support:
            bound_list_5.append(key)
    return [bound_list_1, bound_list_2, bound_list_3, bound_list_4, bound_list_5]


def writeFeatureData(output_file, signalFeature, DIFeature, Pvalue_feature):
    signal_keys = set(signalFeature.keys())
    DI_keys = set(DIFeature.keys())
    Pvalue_keys = set(Pvalue_feature.keys())
    key_intersection = sorted(list(signal_keys.intersection(DI_keys, Pvalue_keys)), key=lambda x: x[0])
    with open(output_file, mode="a") as file:
        for bound in key_intersection:
            file.write(str(bound[0]) + '\t' + str(bound[1]) + '\t')
            for value in signalFeature[bound]:
                file.write(str(value) + '\t')
            for value in DIFeature[bound]:
                file.write(str(value) + '\t')
            for value in Pvalue_feature[bound]:
                file.write(str(value) + '\t')
            file.write('\n')


def getFeatureForModel(matrix, bound_with_info, min_info_support, binSignal_window_size_list, DI_window_size_list, P_window_size_list):
    bound_lists = getFivePartBounds(bound_with_info, min_info_support)
    test_loader_list = []
    for i in range(len(bound_lists)):
        boundToSignalFeature, boundToDIFeature, boundToPvalueFeature = getFinalFeatureForPart(bound_lists[i], matrix, binSignal_window_size_list, DI_window_size_list, P_window_size_list)
        feature_file = args.output + "test_feature/" + str(i + 1) + ".txt"
        writeFeatureData(feature_file, boundToSignalFeature, boundToDIFeature, boundToPvalueFeature)
        test_loader_list.append(getTestLoaderForBounds(file=feature_file, batch_size=batch_size))
    remove_folder(args.output + 'test_feature')
    return test_loader_list


def getSubMatrix(mat, left_bin, right_bin):
    matrix = []
    for i in range(left_bin-1, right_bin):
        row_temp = []
        for j in range(left_bin-1, right_bin):
            row_temp.append(mat[i][j])
        matrix.append(row_temp)
    matrix = np.array(matrix)
    if matrix.shape[0] >= 3:
        if np.sum(np.where(np.diagonal(matrix), 0, 1)) == 0:
            with open(args.output + "sub_matrix/" + str(left_bin) + "_" + str(right_bin) + ".txt", 'w') as f:
                for i in range(matrix.shape[0]):
                    for j in range(i + 1, matrix.shape[1]):
                        f.write(str(i + left_bin) + '\t' + str(j + left_bin) + '\t' + str(matrix[i][j]) + '\n')
            return 1
        return 0
    else:
        return 0


def read_Infomap_module_num(file):
    try:
        with open(file, 'r') as f:
            for line in f.readlines():
                newline = line.rstrip('\n').split(' ')
                if newline[1] == "partitioned":
                    module_num = int(newline[6])
        return module_num
    except FileNotFoundError:
        pass


def extract_consecutive_numbers(lst):
    consecutive_sequences = []
    sequence = []

    for num in lst:
        if not sequence or num == sequence[-1] + 1:
            sequence.append(num)
        else:
            if len(sequence) >= 3:
                consecutive_sequences.append(sequence)
            sequence = [num]

    # 检查最后一个序列
    if len(sequence) >= 3:
        consecutive_sequences.append(sequence)

    return consecutive_sequences


def getFilteredDualBounds(filtered_dual_bounds, bound_with_pv):
    dual_bounds_temp = filtered_dual_bounds[:]
    iterBoundsList = []
    for i in range(len(dual_bounds_temp)):
        for j in range(i + 1, len(dual_bounds_temp)):
            set1 = set(dual_bounds_temp[i])
            set2 = set(dual_bounds_temp[j])
            if set1.intersection(set2):
                iterBoundsList = sorted(list(set(iterBoundsList).union(set1.union(set2))))
    consecutive_sequences = extract_consecutive_numbers(iterBoundsList)
    for seq in consecutive_sequences:
        if len(seq) == 3:
            left_bound = (seq[0], seq[1])
            right_bound = (seq[1], seq[2])
            if bound_with_pv[left_bound] > bound_with_pv[right_bound]:
                filtered_dual_bounds.remove(seq[1:])
            else:
                filtered_dual_bounds.remove(seq[:2])
        elif len(seq) == 4:
            pvList = []
            for i in range(3):
                pvList.append(bound_with_pv[(seq[i], seq[i + 1])])
            max_index = pvList.index(max(pvList))
            for i in range(3):
                if i != max_index:
                    filtered_dual_bounds.remove(seq[i:i + 2])
        elif len(seq) > 4:
            for i in range(1, len(seq) - 2):
                if seq[i:i + 2] in filtered_dual_bounds:
                    filtered_dual_bounds.remove(seq[i:i + 2])
    filtered_dual_bounds = np.array(sorted(filtered_dual_bounds, key=lambda x: x[0]))
    return filtered_dual_bounds


def getTestLoaderForBounds(file, batch_size):
    test_data = BoundaryDataSet(file)
    if test_data.n_samples != 0:
        return DataLoader(test_data, batch_size=batch_size, shuffle=False)
    else:
        return []


def getPartBounds(test_loader, bound_with_pv, thresholds, part_num):
    bound_with_pv_model = dict()
    pv_model = []
    with torch.no_grad():
        for k, (features, labels) in enumerate(test_loader):
            bounds = features[:, :2].numpy()
            features = features[:, 2:].to(device)
            outputs = model(features).numpy()
            for i in range(len(outputs)):
                dual_bound = (int(bounds[i][0]), int(bounds[i][1]))
                bound_with_pv[dual_bound] = outputs[i]
                bound_with_pv_model[dual_bound] = outputs[i]
                pv_model.append(outputs[i].item())
    threshold = np.percentile(pv_model, thresholds[part_num - 1])
    bounds_from_model = []
    for bound in bound_with_pv_model:
        if bound_with_pv_model[bound] >= threshold:
            bounds_from_model.append(bound)
    return bounds_from_model


def getPartBoundsForZero(test_loader, bound_with_pv, thresholds):
    bound_with_pv_model = dict()
    pv_model = []
    with torch.no_grad():
        for k, (features, labels) in enumerate(test_loader):
            bounds = features[:, :2].numpy()
            features = features[:, 2:].to(device)
            outputs = model(features).numpy()
            for i in range(len(outputs)):
                dual_bound = (int(bounds[i][0]), int(bounds[i][1]))
                bound_with_pv[dual_bound] = outputs[i]
                bound_with_pv_model[dual_bound] = outputs[i]
                pv_model.append(outputs[i].item())
    bounds_from_model = []
    for bound in bound_with_pv_model:
        if bound_with_pv_model[bound] >= thresholds[0]:
            bounds_from_model.append(bound)
    return bounds_from_model


def getFilteredBounds(test_loaders):
    bound_with_pv = dict()
    bounds_from_part = []
    for i in range(1, 6):
        if i == 1:
            bounds_from_part.append(
                getPartBoundsForZero(test_loader=test_loaders[i-1], bound_with_pv=bound_with_pv, thresholds=thresholds))
        else:
            if test_loaders[i - 1]:
                bounds_from_part.append(
                    getPartBounds(test_loader=test_loaders[i-1], bound_with_pv=bound_with_pv, thresholds=thresholds, part_num=i))
    bounds_from_model = []
    for bounds in bounds_from_part:
        bounds_from_model += bounds
    return bound_with_pv, sorted(bounds_from_model, key=lambda x: x[0])


def writeBoundsWithInfoAndPredictionValue(file):
    with open(file, 'w') as file:
        for b in bounds_from_model:
            file.write(str(b[0]) + '\t' + str(b[1]) + '\t' + str(bound_with_info[b]) + '\t' + str(bound_with_pv[b]) + '\n')


def getFinalFeatureForPart(bound_list, matrix, binSignal_window_size_list, DI_window_size_list, P_window_size_list):
    # Get bounds' signal features;
    signal_feature = dict()
    for binSignal_window_size in binSignal_window_size_list:
        signal_feature_for_window = getSignalFeature(bound_list, matrix, binSignal_window_size, binSignal_size)
        signal_feature = merge_feature_for_window(signal_feature_for_window, signal_feature)
    signal_feature = filter_feature(signal_feature)

    # Get bounds' DI features;
    DI_feature = dict()
    for DI_window_size in DI_window_size_list:
        DI_feature_for_window = getDIFeature(bound_list, matrix, DI_window_size, DI_size)
        DI_feature = merge_feature_for_window(DI_feature_for_window, DI_feature)
    DI_feature = filter_feature(DI_feature)

    # Get Wilcox Rank Sum test feature;
    Pvalue_feature = dict()
    for P_window_size in P_window_size_list:
        Pvalue_feature_fow_window = getPvalueFeature(bound_list, matrix, P_window_size)
        Pvalue_feature = merge_feature_for_window(Pvalue_feature_fow_window, Pvalue_feature)
    Pvalue_feature = filter_feature(Pvalue_feature)
    return signal_feature, DI_feature, Pvalue_feature


def merge_feature_for_window(feature_for_window, feature):
    if len(feature) > 0:
        new_signal_feature = dict()
        keys1 = set(feature_for_window.keys())
        keys2 = set(feature.keys())
        intersection_keys = keys1 & keys2
        for key in intersection_keys:
            new_signal_feature[key] = feature[key] + feature_for_window[key]
        feature = new_signal_feature
    else:
        feature = feature_for_window
    return feature


def filter_feature(feature):
    keys_to_remove = []
    for key in feature.keys():
        num_zeros = feature[key].count(0)
        if (num_zeros / len(feature[key])) * 100 > 80:
            keys_to_remove.append(key)
    return {key: value for key, value in feature.items() if key not in keys_to_remove}


def generate_TAD():
    TAD = []
    for i in range(len(left_bounds)):
        anchor = left_bounds[i]
        candidate_bounds = [x for x in right_bounds if x < anchor and 2 < (anchor - x)]
        if i == 0 and start_bin not in candidate_bounds:
            candidate_bounds.insert(0, start_bin)
        candidate_bounds = sorted(candidate_bounds, reverse=True)
        if len(candidate_bounds):
            for candidate in candidate_bounds:
                flag = getSubMatrix(matrix, candidate, anchor)
                if flag == 1:
                    subprocess.call(
                        [sys.path[0] + "/Infomap",
                         args.output + "sub_matrix/" + str(candidate) + "_" + str(anchor) + ".txt",
                         args.output + "sub_matrix", "-N", "20", "--silent"])
                    if read_Infomap_module_num(args.output + "sub_matrix/" + str(candidate) + "_" + str(anchor) + ".tree") == 1:
                        TAD.append([candidate, anchor])
                    else:
                        break
                else:
                    break
    return TAD


def get_TAD_level(TAD):
    dict_tad_level_chr = dict()
    for i in range(len(TAD)):
        multi_level_i = [[TAD[i]]]
        n = 0
        flag = True
        while flag:
            upper_level = []
            for k in multi_level_i[n]:
                for j in range(len(TAD)):
                    if k != TAD[j]:
                        if int(k[0]) >= int(TAD[j][0]) and int(k[1]) <= int(TAD[j][1]):
                            upper_level.append(TAD[j])
            if len(upper_level) != 0:
                multi_level_i.append(upper_level)
                n += 1
            else:
                flag = False
        dict_tad_level_chr[(TAD[i][0], TAD[i][1])] = len(multi_level_i) - 1
    return dict_tad_level_chr


def find_gaps(tad_level, start, end):
    tads_0 = sorted([tad for tad in tad_level.keys() if tad_level[tad] == 0], key=lambda x: x[0])
    gaps = []
    for tad in tads_0:
        if start < tad[0]:
            gaps.append([start, tad[0]])
            start = tad[1]
        else:
            start = tad[1]
    if start < end:
        gaps.append([start, end])
    for gap in gaps:
        tad_level[(gap[0], gap[1])] = -1
    tads = sorted([(tad[0], tad[1]) for tad in list(tad_level.keys()) + gaps], key=lambda x: x[0])
    tads = sorted([x for x in set(tuple(x) for x in tads)], key=lambda x: x[0])
    return tad_level, tads


def logo():
    print( """* ============================================= *
*     ____ _____ _   _ _____  ______ _____      *
*    |  _ \_   _| \ | |  __ \|  ____|  __ \     *
*    | |_) || | |  \| | |  | | |__  | |__) |    *
*    |  _ < | | | . ` | |  | |  __| |  _  /     *
*    | |_) || |_| |\  | |__| | |____| | \ \     *
*    |____/_____|_| \_|_____/|______|_|  \_\    *""")


def delimiter():
    print("*                                               *" )
    print("* ============================================= *" )
    print("*                                               *" )


def makedirs(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def remove_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


if __name__ == '__main__':

    device = torch.device('cpu')
    thresholds = [0.77, 85, 80, 30, 0]
    batch_size = 500
    learning_rate = 0.001
    min_info_support = 8
    binSignal_size = 5
    DI_size = 5
    binSignal_window_size_list = [5, 10, 20, 30, 50, 60, 70, 80, 90, 100]
    DI_window_size_list = [5, 10, 20, 30, 50, 60, 70, 80, 90, 100]
    P_window_size_list = [5, 10, 20, 30, 50, 60, 70, 80, 90, 100]

    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', '-m', help='Hi-C contact matrix (N x N)', type=str)
    parser.add_argument('--resolution', '-r', help='resolution of Hi-C matrix (kb)', type=int)
    parser.add_argument('--chromosome', '-chr', help='chromosome of Hi-C matrix', type=str)
    parser.add_argument('--normalization', '-n', help='normalization method', default='SCN', type=str)
    parser.add_argument('--output', '-o', help='path_to_output', default=sys.path[0]+'/BINDER_result/', type=str)
    args = parser.parse_args()

    if not args.matrix or not args.resolution:
        parser.print_help()
        sys.exit("Error: You must provide both 'matrix' and 'resolution' arguments.")

    logo()
    delimiter()

    makedirs(args.output[:-1])

    matrix = np.loadtxt(args.matrix)

    print("*               Normalizing matrix              *")
    print("*                                               *")
    print("*                                               *")

    if args.normalization == 'SCN':
        matrix = function.SCN(matrix)
        model = torch.load(sys.path[0] + "/model/model_SCN.pkl").to(device)

    if args.normalization == 'KR':
        matrix, _, _ = function.KR(matrix)
        model = torch.load(sys.path[0] + "/model/model_KR.pkl").to(device)

    if args.normalization == 'ICE':
        matrix = function.ICE(matrix)
        model = torch.load(sys.path[0] + "/model/model_ICE.pkl").to(device)

    if args.normalization == 'sqrtVC':
        matrix = function.SQRTVCnorm(matrix)
        model = torch.load(sys.path[0] + "/model/model_sqrtVC.pkl").to(device)

    print("*               Normalization done              *")
    print("*                                               *")
    print("* ============================================= *")
    print("*                                               *")
    print("*            Generating TAD boundaries          *")
    print("*                                               *")
    print("* ============================================= *")

    makedirs(args.output + 'map')

    start_bin, end_bin, bound_with_info = get_Infomap_bounds(matrix=matrix,
                                            map_path=args.output + '/map/',
                                            length=int(2000 / args.resolution),
                                            lap=10)

    remove_folder(args.output + 'map')

    makedirs(args.output + 'test_feature')

    loaders = getFeatureForModel(matrix=matrix, bound_with_info=bound_with_info,
                                 min_info_support=min_info_support,
                                 binSignal_window_size_list=binSignal_window_size_list,
                                 DI_window_size_list=DI_window_size_list,
                                 P_window_size_list=P_window_size_list)

    remove_folder(args.output + 'test_feature')

    # model = torch.load(sys.path[0] + "/model.pkl").to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bound_with_pv, bounds_from_model = getFilteredBounds(test_loaders=loaders)

    filtered_dual_bounds = []
    for bound in bounds_from_model:
        filtered_dual_bounds.append([bound[0], bound[1]])
    filtered_dual_bounds = getFilteredDualBounds(filtered_dual_bounds=filtered_dual_bounds, bound_with_pv=bound_with_pv)

    print("*                                               *")
    print("*         TAD boundaries are generated          *")
    print("*                                               *")
    print("* ============================================= *")
    print("*                                               *")
    print("*                Generating TADs                *")
    print("*                                               *")

    left_bounds = filtered_dual_bounds[:, 0].tolist()
    left_bounds.append(end_bin)
    right_bounds = filtered_dual_bounds[:, 1].tolist()
    right_bounds.insert(0, start_bin)

    makedirs(args.output + 'sub_matrix')

    TAD = generate_TAD()

    remove_folder(args.output + 'sub_matrix')

    TAD = sorted(TAD, key=lambda x: x[0])
    TAD_base = []
    for tad in TAD:
        TAD_base.append([(tad[0] - 1) * args.resolution * 1000, tad[1] * args.resolution * 1000])
    tad_level, TAD = find_gaps(get_TAD_level(TAD_base), 0, matrix.shape[0] * args.resolution * 1000)

    with open(args.output + 'Result.' + args.chromosome, 'w') as file:
        file.write("Left_position" + '\t' + "Right_position" + '\t' + "Level" + '\t' + 'Type' + '\n')
        for tad in TAD:
            level = tad_level[tad]
            if level != -1:
                file.write(str(tad[0]) + '\t' + str(tad[1]) + '\t' + str(tad_level[tad]) + '\t' + 'domain' + '\n')
            else:
                file.write(str(tad[0]) + '\t' + str(tad[1]) + '\t' + 'non-level' + '\t' + 'gap' + '\n')


    print("* ============================================= *")
    print("*                                               *")
    print("*               TADs are generated              *")
    print("*                                               *")
    print("* ============================================= *")


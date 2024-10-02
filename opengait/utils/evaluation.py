import os
from time import strftime, localtime
import torch
import numpy as np
import torch.nn.functional as F
from utils import get_msg_mgr, mkdir, MeanIOU


def log_or_print(msg):
    try:
        msg_mgr = get_msg_mgr()
        msg_mgr.log_info(msg)
    except:
        print(msg)


def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin


def __find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=1e-8):
    assert len(score_vec.shape) == 1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == bool
    score_neg = score_vec[~label_vec]
    score_neg = np.sort(score_neg)[::-1]  # score from high to low
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        epsilon = 1e-5
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0] + epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1] - epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = (num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm == 0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm - 1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds


def compute_dir_far(score_mat, label_mat, ranks=[20], FARs=[1.0], get_retrievals=False):
    """ Closed/Open-set Identification.
        A general case of Cummulative Match Characteristic (CMC)
        where thresholding is allowed for open-set identification.
    args:
        score_mat:            a P x G matrix, P is number of probes, G is size of gallery
        label_mat:            a P x G matrix, bool
        ranks:                a list of integers
        FARs:                 false alarm rates, if 1.0, closed-set identification (CMC)
        get_retrievals:       not implemented yet
    return:
        DIRs:                 an F x R matrix, F is the number of FARs, R is the number of ranks,
                              flatten into a vector if F=1 or R=1.
        FARs:                 an vector of length = F.
        thredholds:           an vector of length = F.
    """
    assert score_mat.shape == label_mat.shape
    assert np.all(label_mat.astype(np.float32).sum(axis=1) <= 1)
    # Split the matrix for match probes and non-match probes
    # subfix _m: match, _nm: non-match
    # For closed set, we only use the match probes
    mate_indices = label_mat.astype(bool).any(axis=1)
    score_mat_m = score_mat[mate_indices, :]
    label_mat_m = label_mat[mate_indices, :]
    score_mat_nm = score_mat[np.logical_not(mate_indices), :]
    mate_indices = np.argwhere(mate_indices).flatten()

    # print('mate probes: %d, non mate probes: %d' % (score_mat_m.shape[0], score_mat_nm.shape[0]))

    # Find the thresholds for different FARs
    max_score_nm = np.max(score_mat_nm, axis=1)
    label_temp = np.zeros(max_score_nm.shape, dtype=bool)
    if len(FARs) == 1 and FARs[0] >= 1.0:
        # If only testing closed-set identification, use the minimum score as np.vstack(thresesholds)
        # in case there is no non-mate probes
        thresholds = [np.min(score_mat) - 1e-10]
    else:
        # If there is open-set identification, find the thresholds by FARs.
        assert score_mat_nm.shape[
                   0] > 0, "For open-set identification (FAR<1.0), there should be at least one non-mate probe!"
        thresholds = __find_thresholds_by_FAR(max_score_nm, label_temp, FARs=FARs)

    # Sort the labels row by row according to scores
    sort_idx_mat_m = np.argsort(score_mat_m, axis=1)[:, ::-1]
    sorted_label_mat_m = np.ndarray(label_mat_m.shape, dtype=bool)
    sorted_score_mat_m = score_mat_m.copy()
    for row in range(label_mat_m.shape[0]):
        sort_idx = (sort_idx_mat_m[row, :])
        sorted_label_mat_m[row, :] = label_mat_m[row, sort_idx]
        sorted_score_mat_m[row, :] = score_mat_m[row, sort_idx]

    # Calculate DIRs for different FARs and ranks
    gt_score_m = score_mat_m[label_mat_m]
    assert gt_score_m.size == score_mat_m.shape[0]

    DIRs = np.zeros([len(FARs), len(ranks)], dtype=np.float32)
    FARs = np.zeros([len(FARs)], dtype=np.float32)
    success = np.ndarray((len(FARs), len(ranks)), dtype=object)
    for i, threshold in enumerate(thresholds):
        for j, rank in enumerate(ranks):
            score_rank = gt_score_m >= threshold
            retrieval_rank = sorted_label_mat_m[:, 0:rank].any(axis=1)
            DIRs[i, j] = (score_rank & retrieval_rank).astype(np.float32).mean()
            if get_retrievals:
                success[i, j] = (score_rank & retrieval_rank)
        if score_mat_nm.shape[0] > 0:
            FARs[i] = (max_score_nm >= threshold).astype(np.float32).mean()

    if DIRs.shape[0] == 1 or DIRs.shape[1] == 1:
        DIRs = DIRs.flatten()
        success = success.flatten()

    if get_retrievals:
        return DIRs, FARs, thresholds, mate_indices, success, sort_idx_mat_m, sorted_score_mat_m

    return DIRs, FARs, thresholds


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result

# Modified From https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/evaluator.py


def identification(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    # sample_num = len(feature)

    probe_seq_dict = {'CASIA-B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}

    gallery_seq_dict = {'CASIA-B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]),
                    view_num, view_num, num_rank]) - 1.
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                        view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                        view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x, metric)
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)
    result_dict = {}
    np.set_printoptions(precision=3, suppress=True)
    if 'OUMVLP' not in dataset:
        for i in range(1):
            log_or_print(
                '===Rank-%d (Include identical-view cases)===' % (i + 1))
            log_or_print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(acc[0, :, :, i]),
                np.mean(acc[1, :, :, i]),
                np.mean(acc[2, :, :, i])))
        for i in range(1):
            log_or_print(
                '===Rank-%d (Exclude identical-view cases)===' % (i + 1))
            log_or_print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                de_diag(acc[0, :, :, i]),
                de_diag(acc[1, :, :, i]),
                de_diag(acc[2, :, :, i])))
        result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, i])
        result_dict["scalar/test_accuracy/BG"] = de_diag(acc[1, :, :, i])
        result_dict["scalar/test_accuracy/CL"] = de_diag(acc[2, :, :, i])
        np.set_printoptions(precision=2, floatmode='fixed')
        for i in range(1):
            log_or_print(
                '===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
            log_or_print('NM: {}'.format(de_diag(acc[0, :, :, i], True)))
            log_or_print('BG: {}'.format(de_diag(acc[1, :, :, i], True)))
            log_or_print('CL: {}'.format(de_diag(acc[2, :, :, i], True)))
    else:
        log_or_print('===Rank-1 (Include identical-view cases)===')
        log_or_print('NM: %.3f ' % (np.mean(acc[0, :, :, 0])))
        log_or_print('===Rank-1 (Exclude identical-view cases)===')
        log_or_print('NM: %.3f ' % (de_diag(acc[0, :, :, 0])))
        log_or_print(
            '===Rank-1 of each angle (Exclude identical-view cases)===')
        log_or_print('NM: {}'.format(de_diag(acc[0, :, :, 0], True)))
        result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, 0])
    return result_dict


def identification_real_scene_closed_set(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)

    gallery_seq_type = {'0001-1000': ['1', '2'],
                        "HID2021": ['0'], '0001-1000-test': ['0'],
                        'GREW': ['01'], 'TTG-200': ['1']}
    probe_seq_type = {'0001-1000': ['3', '4', '5', '6'],
                      "HID2021": ['1'], '0001-1000-test': ['1'],
                      'GREW': ['02'], 'TTG-200': ['2', '3', '4', '5', '6']}

    num_rank = 20
    acc = np.zeros([num_rank]) - 1.
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = label[pseq_mask]

    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.cpu().sort(1)[1].numpy()
    acc = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                          0) * 100 / dist.shape[0], 2)
    log_or_print('==Rank-1==')
    log_or_print('%.3f' % (np.mean(acc[0])))
    log_or_print('==Rank-5==')
    log_or_print('%.3f' % (np.mean(acc[4])))
    log_or_print('==Rank-10==')
    log_or_print('%.3f' % (np.mean(acc[9])))
    log_or_print('==Rank-20==')
    log_or_print('%.3f' % (np.mean(acc[19])))
    return {"scalar/test_accuracy/Rank-1": np.mean(acc[0]), "scalar/test_accuracy/Rank-5": np.mean(acc[4])}


def identification_real_scene_open_set(data, dataset, metric='euc'):
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)

    gallery_seq_type = {'0001-1000': ['1', '2'],
                        "HID2021": ['0'], '0001-1000-test': ['0'],
                        'GREW': ['01'], 'TTG-200': ['1']}
    probe_seq_type = {'0001-1000': ['3', '4', '5', '6'],
                      "HID2021": ['1'], '0001-1000-test': ['1'],
                      'GREW': ['02'], 'TTG-200': ['2', '3', '4', '5', '6']}
    open_set_protocols = {'GREW': 'datasets/GREW/openset_nonmated.npy'}

    gallery_ids = np.asarray(sorted(list(set(label))))
    open_set_protocols = np.load(open_set_protocols[dataset])
    tpirs = []
    for protocol in open_set_protocols:
        open_set_gallery_ids = list(gallery_ids[protocol])
        gseq_mask = np.isin(seq_type, gallery_seq_type[dataset]) & np.isin(label, open_set_gallery_ids)
        gallery_x = feature[gseq_mask, :]
        gallery_y = label[gseq_mask]
        pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
        probe_x = feature[pseq_mask, :]
        probe_y = label[pseq_mask]

        dist = cuda_dist(probe_x, gallery_x, metric).cpu().numpy()
        gt = probe_y[:, None] == gallery_y[None, :]
        tpir, _, _ = compute_dir_far(1 / (1 + dist), gt, FARs=[0.01])
        tpirs.append(tpir[0] * 100)
    log_or_print('==Mean TPIR@1%FPIR==')
    log_or_print('%.2f' % np.mean(tpirs))
    log_or_print('==Median TPIR@1%FPIR==')
    log_or_print('%.2f' % np.median(tpirs))
    log_or_print('==STD TPIR@1%FPIR==')
    log_or_print('%.2f' % np.std(tpirs))
    return {"scalar/test_accuracy/Mean_TPIR@1%FPIR": np.mean(tpirs),
            "scalar/test_accuracy/Median_TPIR@1%FPIR": np.median(tpirs),
            "scalar/test_accuracy/STD_TPIR@1%FPIR": np.std(tpirs)}


def identification_real_scene(data, dataset, metric='euc', save_embeddings=False):
    if save_embeddings:
        import torch
        torch.save(data, f'{save_embeddings}.pth')

    closed_set_results = identification_real_scene_closed_set(data, dataset, metric)
    open_set_results = identification_real_scene_open_set(data, dataset, metric)
    return {**closed_set_results, **open_set_results}


def identification_GREW_submission(data, dataset, metric='euc'):
    get_msg_mgr().log_info("Evaluating GREW")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)
    gallery_seq_type = {'GREW': ['01', '02']}
    probe_seq_type = {'GREW': ['03']}
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = view[pseq_mask]

    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.cpu().sort(1)[1].numpy()

    save_path = os.path.join(
        "GREW_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")
    mkdir("GREW_result")
    with open(save_path, "w") as f:
        f.write("videoId,rank1,rank2,rank3,rank4,rank5,rank6,rank7,rank8,rank9,rank10,rank11,rank12,rank13,rank14,rank15,rank16,rank17,rank18,rank19,rank20\n")
        for i in range(len(idx)):
            r_format = [int(idx) for idx in gallery_y[idx[i, 0:20]]]
            output_row = '{}'+',{}'*20+'\n'
            f.write(output_row.format(probe_y[i], *r_format))
        print("GREW result saved to {}/{}".format(os.getcwd(), save_path))
    return


def evaluate_HID(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    log_or_print("Evaluating HID")
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)
    seq_type = np.array(seq_type)
    probe_mask = (label == "probe")
    gallery_mask = (label != "probe")
    gallery_x = feature[gallery_mask, :]
    gallery_y = label[gallery_mask]
    probe_x = feature[probe_mask, :]
    probe_y = seq_type[probe_mask]

    feat = np.concatenate([probe_x, gallery_x])
    dist = cuda_dist(feat, feat, metric).cpu().numpy()
    log_or_print("Starting Re-ranking")
    re_rank = re_ranking(dist, probe_x.shape[0], k1=6, k2=6, lambda_value=0.3)
    idx = np.argsort(re_rank, axis=1)

    save_path = os.path.join(
        "HID_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")
    mkdir("HID_result")
    with open(save_path, "w") as f:
        f.write("videoID,label\n")
        for i in range(len(idx)):
            f.write("{},{}\n".format(probe_y[i], gallery_y[idx[i, 0]]))
        print("HID result saved to {}/{}".format(os.getcwd(), save_path))
    return


def re_ranking(original_dist, query_num, k1, k2, lambda_value):
    # Modified from https://github.com/michuanhaohao/reid-strong-baseline/blob/master/utils/re_ranking.py
    all_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(
                np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                            :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + \
        original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def mean_iou(data, dataset):
    labels = data['mask']
    pred = data['pred']
    miou = MeanIOU(pred, labels)
    get_msg_mgr().log_info('mIOU: %.3f' % (miou.mean()))
    return {"scalar/test_accuracy/mIOU": miou}


def evaluate_rank(distmat, p_lbls, g_lbls, max_rank=50):
    '''
    Copy from https://github.com/Gait3D/Gait3D-Benchmark/blob/72beab994c137b902d826f4b9f9e95b107bebd78/lib/utils/rank.py#L12-L63
    '''
    num_p, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)

    matches = (g_lbls[indices] == p_lbls[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each probe
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_p = 0.  # number of valid probe

    for p_idx in range(num_p):
        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[p_idx]
        if not np.any(raw_cmc):
            # this condition is true when probe identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)    # 返回坐标，此处raw_cmc为一维矩阵，所以返回相当于index
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_p += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        pos_idx = np.where(raw_cmc == 1)    # 返回坐标，此处raw_cmc为一维矩阵，所以返回相当于index
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_p += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_p > 0, 'Error: all probe identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_p

    return all_cmc, all_AP, all_INP


def evaluate_Gait3D_closed_set(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    features, labels, cams, time_seqs = data['embeddings'], data['labels'], data['types'], data['views']
    import json
    probe_sets = json.load(open(f'datasets/Gait3D/{dataset}.json', 'rb'))['PROBE_SET']
    probe_mask = []
    for id, ty, sq in zip(labels, cams, time_seqs):
        if '-'.join([id, ty, sq]) in probe_sets:
            probe_mask.append(True)
        else:
            probe_mask.append(False)
    probe_mask = np.array(probe_mask)

    # probe_features = features[:probe_num]
    probe_features = features[probe_mask]
    # gallery_features = features[probe_num:]
    gallery_features = features[~probe_mask]
    # probe_lbls = np.asarray(labels[:probe_num])
    # gallery_lbls = np.asarray(labels[probe_num:])
    probe_lbls = np.asarray(labels)[probe_mask]
    gallery_lbls = np.asarray(labels)[~probe_mask]

    results = {}
    log_or_print(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()
    cmc, all_AP, all_INP = evaluate_rank(dist, probe_lbls, gallery_lbls)

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['scalar/test_accuracy/Rank-{}'.format(r)] = cmc[r - 1] * 100
    results['scalar/test_accuracy/mAP'] = mAP * 100
    results['scalar/test_accuracy/mINP'] = mINP * 100

    # print_csv_format(dataset_name, results)
    log_or_print(results)
    return results


def evaluate_Gait3D_open_set(data, dataset, metric='euc', remove_nonmated_gallery=True, plot_score_dist=False):
    from collections import defaultdict
    import json

    features, labels, cams, time_seqs = data['embeddings'], data['labels'], data['types'], data['views']
    gait3d_protocol = json.load(open(f'datasets/Gait3D/{dataset}.json', 'rb'))
    gallery_ids = np.asarray(gait3d_protocol['TEST_SET'])
    probe_set = gait3d_protocol['PROBE_SET']

    # Initialize dictionaries to store gallery and probe features
    gallery_features = defaultdict(list)
    probe_features = []
    probe_lbls = []
    for feature, label, cam, time_seq in zip(list(features), labels, cams, time_seqs):
        if f'{label}-{cam}-{time_seq}' in probe_set:
            # Sequence is a probe, assign it to probe_features dictionary
            probe_features.append(feature)
            probe_lbls.append(label)
        elif label in gallery_ids:
            # Sequence is in the gallery set, assign it to gallery_features dictionary
            gallery_features[label].append(feature)

    probe_features = np.stack(probe_features)
    probe_lbls = np.asarray(probe_lbls)

    # Aggregate gallery features by averaging
    for label, feats in gallery_features.items():
        gallery_features[label] = [np.mean(np.stack(feats), axis=0)]
    gallery_lbls = np.asarray(list(gallery_features.keys()))
    gallery_features = np.stack([feat[0] for feat in gallery_features.values()])

    log_or_print(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()
    gt = probe_lbls[:, None] == gallery_lbls[None, :]

    # define 50 mated/nonmated splits
    splits = np.load('datasets/Gait3D/openset_nonmated.npy')

    from tqdm import tqdm
    results, tpirs = {}, []
    for split in tqdm(splits, desc='Splits'):
        if remove_nonmated_gallery:
            split_gallery_ids = set(gallery_ids).difference(set(gallery_ids[split]))
        else:
            split_gallery_ids = set(gait3d_protocol['TEST_SET'])

        # select from the distance matrix
        gallery_mask = np.isin(gallery_lbls, list(split_gallery_ids))
        dist_split = dist[:, gallery_mask]
        gt_split = gt[:, gallery_mask]

        tpir, fpir, threshold = compute_dir_far(1 / (1 + dist_split), gt_split, FARs=[0.01])
        tpirs.extend(tpir)

    results['scalar/test_accuracy/Mean_TPIR@1%FPIR'] = np.round(np.mean(tpirs) * 100, 2)
    results['scalar/test_accuracy/STD_TPIR@1%FPIR'] = np.round(np.std(tpirs) * 100, 2)
    results['scalar/test_accuracy/Median_TPIR@1%FPIR'] = np.round(np.median(tpirs) * 100, 2)
    

    if plot_score_dist:
        from matplotlib import pyplot as plt

        # get match scores
        scores = np.copy(1 / (1 + dist))
        scores_match = scores[gt == 1]
        # get nonmatch scores
        scores_nm = scores[gt == 0]
        scores[gt == 1] = -np.inf
        scores_nm_max_per_probe = np.max(scores, axis=1)

        # estimate 1% FPIR threshold
        sorted_scores = np.sort(scores_nm_max_per_probe)
        num_scores_nm = scores_nm_max_per_probe.shape[0]
        threshold = sorted_scores[-int(np.ceil(0.01 * num_scores_nm))]
        log_or_print(f'Threshold: {threshold}')
        log_or_print(f'TPIR: {np.sum(scores_match > threshold)} / {scores_match.shape}')

        zero = scores_nm.min()
        one = scores_match.max() - scores_nm.min()

        scores_match = (scores_match - zero) / one
        scores_nm = (scores_nm - zero) / one
        scores_nm_max_per_probe = (scores_nm_max_per_probe - zero) / one
        threshold = (threshold - zero) / one

        log_or_print(f'Zero: {zero}, One: {one}, Threshold: {threshold}')

        plt.figure(figsize=(16, 4))
        plt.hist(scores_match, bins=20, color='red', alpha=0.5, label='Matches', density=True)
        plt.hist(scores_nm, bins=20, color='blue', alpha=0.5, label='Non-Matches', density=True)
        plt.hist(scores_nm_max_per_probe, bins=20, color='green', alpha=0.5,
                 label='Non-Matches (Max/template)', density=True)
        plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig('score_dist_match_vs_nonmatch.png', bbox_inches='tight')
    log_or_print(results)
    return results


def evaluate_Gait3D(data, dataset, metric='euc', remove_nonmated_gallery=True, plot_score_dist=False, save_embeddings=False):
    if save_embeddings:
        import torch
        torch.save(data, f'{save_embeddings}.pth')

    closed_set_results = evaluate_Gait3D_closed_set(data, dataset, metric=metric)
    open_set_results = evaluate_Gait3D_open_set(data, dataset, metric=metric,
                                                remove_nonmated_gallery=remove_nonmated_gallery,
                                                plot_score_dist=plot_score_dist)
    return {**closed_set_results, **open_set_results}


def evaluate_many(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)   # 对应位置变成从小到大的序号
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(
        np.int32)  # 根据indices调整顺序 g_pids[indices]
    # print(matches)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP


def evaluate_CCPG(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']

    label = np.array(label)
    for i in range(len(view)):
        view[i] = view[i].split("_")[0]
    view_np = np.array(view)
    view_list = list(set(view))
    view_list.sort()

    view_num = len(view_list)

    probe_seq_dict = {'CCPG': [["U0_D0_BG", "U0_D0"], [
        "U3_D3"], ["U1_D0"], ["U0_D0_BG"]]}

    gallery_seq_dict = {
        'CCPG': [["U1_D1", "U2_D2", "U3_D3"], ["U0_D3"], ["U1_D1"], ["U0_D0"]]}
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]),
                   view_num, view_num, num_rank]) - 1.

    ap_save = []
    cmc_save = []
    minp = []
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        gseq_mask = np.isin(seq_type, gallery_seq)
        gallery_x = feature[gseq_mask, :]
        # print("gallery_x", gallery_x.shape)
        gallery_y = label[gseq_mask]
        gallery_view = view_np[gseq_mask]

        pseq_mask = np.isin(seq_type, probe_seq)
        probe_x = feature[pseq_mask, :]
        probe_y = label[pseq_mask]
        probe_view = view_np[pseq_mask]

        log_or_print(
            ("gallery length", len(gallery_y), gallery_seq, "probe length", len(probe_y), probe_seq))
        distmat = cuda_dist(probe_x, gallery_x, metric).cpu().numpy()
        # cmc, ap = evaluate(distmat, probe_y, gallery_y, probe_view, gallery_view)
        cmc, ap, inp = evaluate_many(
            distmat, probe_y, gallery_y, probe_view, gallery_view)
        ap_save.append(ap)
        cmc_save.append(cmc[0])
        minp.append(inp)

    # print(ap_save, cmc_save)

    log_or_print(
        '===Rank-1 (Exclude identical-view cases for Person Re-Identification)===')
    log_or_print('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
        cmc_save[0]*100, cmc_save[1]*100, cmc_save[2]*100, cmc_save[3]*100))

    log_or_print(
        '===mAP (Exclude identical-view cases for Person Re-Identification)===')
    log_or_print('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
        ap_save[0]*100, ap_save[1]*100, ap_save[2]*100, ap_save[3]*100))

    log_or_print(
        '===mINP (Exclude identical-view cases for Person Re-Identification)===')
    log_or_print('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
                     (minp[0]*100, minp[1]*100, minp[2]*100, minp[3]*100))

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        for (v1, probe_view) in enumerate(view_list):
            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                    view, [gallery_view])
                gallery_x = feature[gseq_mask, :]
                gallery_y = label[gseq_mask]

                pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                    view, [probe_view])
                probe_x = feature[pseq_mask, :]
                probe_y = label[pseq_mask]

                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.sort(1)[1].cpu().numpy()
                # print(p, v1, v2, "\n")
                acc[p, v1, v2, :] = np.round(
                    np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                           0) * 100 / dist.shape[0], 2)
    result_dict = {}
    for i in range(1):
        log_or_print(
            '===Rank-%d (Include identical-view cases)===' % (i + 1))
        log_or_print('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
            np.mean(acc[0, :, :, i]),
            np.mean(acc[1, :, :, i]),
            np.mean(acc[2, :, :, i]),
            np.mean(acc[3, :, :, i])))
    for i in range(1):
        log_or_print(
            '===Rank-%d (Exclude identical-view cases)===' % (i + 1))
        log_or_print('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
            de_diag(acc[0, :, :, i]),
            de_diag(acc[1, :, :, i]),
            de_diag(acc[2, :, :, i]),
            de_diag(acc[3, :, :, i])))
    result_dict["scalar/test_accuracy/CL"] = acc[0, :, :, i]
    result_dict["scalar/test_accuracy/UP"] = acc[1, :, :, i]
    result_dict["scalar/test_accuracy/DN"] = acc[2, :, :, i]
    result_dict["scalar/test_accuracy/BG"] = acc[3, :, :, i]
    np.set_printoptions(precision=2, floatmode='fixed')
    for i in range(1):
        log_or_print(
            '===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
        log_or_print('CL: {}'.format(de_diag(acc[0, :, :, i], True)))
        log_or_print('UP: {}'.format(de_diag(acc[1, :, :, i], True)))
        log_or_print('DN: {}'.format(de_diag(acc[2, :, :, i], True)))
        log_or_print('BG: {}'.format(de_diag(acc[3, :, :, i], True)))
    return result_dict

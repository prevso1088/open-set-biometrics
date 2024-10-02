import torch
from torch import nn
from torch.nn import functional as F

from modeling.losses.base import BaseLoss, gather_and_scale_wrapper


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine)

    def forward(self, x):
        """
            x: [p ...]
        """
        p, *rest = x.shape
        x = x.view(p, -1).permute(1, 0).contiguous()  # [_, p]
        x = self.bn(x)
        x = x.permute(1, 0).contiguous().view(p, *rest)
        return x


class NormedSigmoid(nn.Module):
    def __init__(self, num_parts=16, scale=1., norm=False):
        super(NormedSigmoid, self).__init__()
        self.norm = BatchNorm1d(num_parts, affine=False) if norm else nn.Identity()
        self.scale = scale

    def forward(self, x):
        """
            x: [p, ....]
        """
        return torch.sigmoid(self.scale * self.norm(x))


def compute_distance(x, y):
    """
        x: [p, n_x, c]
        y: [p, n_y, c]
    """
    x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
    y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
    inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
    dist = x2 + y2 - 2 * inner
    dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
    return dist


def find_matches(scores, labels_g, labels_p):
    """
        scores: [p, n_g, n_p]
        labels_g: [n_g]
        labels_p: [n_p]
    """
    # mask indicating matching gallery-probe pairs
    match_mask = labels_g.unsqueeze(1) == labels_p.unsqueeze(0)

    # use the match mask to extract the scores
    return scores[:, None, match_mask]  # Shape: [p, 1, n_matches]


class OpenSetLoss(BaseLoss):
    def __init__(self, num_folds=4, rank=1, score_scale=9.0, rank_scale=1.5, rank_diff_scale=9.0, num_parts=16):
        super(OpenSetLoss, self).__init__()
        self.num_folds = num_folds
        self.rank = rank
        self.num_parts = num_parts

        self.rank_sigmoid = NormedSigmoid(num_parts=num_parts, scale=rank_scale)
        self.thres_sigmoid = NormedSigmoid(num_parts=num_parts, scale=score_scale)
        self.rank_diff_sigmoid = NormedSigmoid(num_parts=num_parts, scale=rank_diff_scale)

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]

        # divide gallery into 4 folds
        unique_labels = torch.unique(labels)
        folds = torch.chunk(unique_labels, self.num_folds)

        dirs = []
        softmax_scores_nm = []
        for fold in folds:
            embeddings_gallery = []
            labels_gallery = []
            probe_match_set = []
            probe_nonmatch_set = []

            for subject in unique_labels:
                # get embeddings and labels for the current subject
                subject_indices = torch.nonzero(labels == subject, as_tuple=False).view(-1)

                if subject in fold:
                    # add everything to probe
                    probe_nonmatch_set.extend(subject_indices.tolist())
                else:
                    # randomly divide the embeddings into gallery and probe sets
                    num_sequences = len(subject_indices)
                    split_index = max(1, int(num_sequences / 2 + .5))
                    embeddings_gallery.append(embeddings[subject_indices[:split_index]].mean(0))
                    labels_gallery.append(subject)
                    probe_match_set.extend(subject_indices[split_index:].tolist())

            # compute gallery-probe distances
            embeddings_gallery = torch.stack(
                embeddings_gallery).permute(2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
            embeddings_probe = embeddings[probe_match_set, :].permute(
                2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
            dist_fold = compute_distance(embeddings_gallery, embeddings_probe)  # [p, n_g, n_p]

            embeddings_nm = embeddings[probe_nonmatch_set, :].permute(
                2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
            dist_nm = compute_distance(embeddings_gallery, embeddings_nm)  # [p, n_g, n_p]

            # convert from distances to scores
            scores_fold = 1 / (1 + dist_fold)
            scores_nm = 1 / (1 + dist_nm)

            # whether each probe's rank is less than r
            labels_gallery = torch.as_tensor(labels_gallery, device=labels.device)
            labels_probe = torch.as_tensor(labels[probe_match_set], device=labels.device)
            scores_match = find_matches(scores_fold, labels_gallery, labels_probe)  # [p, 1, n_matches]
            ranks = self.soft_rank(scores_fold, scores_match)
            ranks = self.rank_sigmoid(self.rank - ranks)

            softmax_scores_nm.append(torch.sum(torch.softmax(-dist_nm, dim=1) * scores_nm, dim=1))  # [p, n_p]

            # use the set of scores as thresholds for FNIR
            thetas = scores_nm.view(scores_nm.shape[0], -1, 1)  # [p, num_thetas, 1]
            thresh = self.thres_sigmoid(scores_match - thetas)
            dirs.append(torch.mean(ranks.unsqueeze(1) * thresh, dim=-1))  # dir: [p, num_thetas, n_p]

        dirs = torch.cat(dirs, dim=1).contiguous().view(self.num_parts, -1)
        softmax_scores_nm = torch.cat(softmax_scores_nm, dim=1).contiguous().view(self.num_parts, -1)
        losses = 1 - dirs.mean(1) + 4 * softmax_scores_nm.mean(1)

        self.info.update({
            'loss': losses.detach().clone(),
            'dirs': dirs.mean(1),
            'nm_scores': softmax_scores_nm.mean(1),
        })

        for name, child in self.named_modules():
            if isinstance(child, BatchNorm1d):
                self.info.update({
                    f'{name}_mean': child.bn.running_mean,
                    f'{name}_std': child.bn.running_var.sqrt(),
                })

        return losses, self.info

    def soft_rank(self, scores, scores_match):  # , scale=6.):
        """
            dist: [p, n_g, n_p]
            labels_g: [n_g]
            labels: [n_p]
        """
        scores_diff = scores - scores_match  # Shape: [p, n_g, n_p]

        # compute the soft rank
        return torch.sum(self.rank_diff_sigmoid(scores_diff), dim=1)  # Shape: [p, n_p]


# Sample Usage
if __name__ == "__main__":
    import random

    n = 128  # Batch size
    c = 256  # Number of channels
    p = 16  # Number of body parts

    # Generate perfect embedding tensor and label tensor
    labels = torch.arange(32).repeat(4)
    embeddings = torch.nn.functional.one_hot(labels, num_classes=c)
    embeddings = embeddings.unsqueeze(-1).repeat((1, 1, p)).float()

    # Generate random embedding tensor and label tensor
    embeddings = torch.randn(n, c, p).cuda()
    labels = torch.randint(0, 32, (n,)).cuda()

    # Parameters
    S1 = 32  # Number of subjects to select
    S2 = 4  # Number of sequences per subject to select

    # Create a list of unique subjects
    unique_subjects = list(set(labels))

    # Select S1 subjects without replacement
    selected_subjects = random.sample(unique_subjects, S1)

    # Initialize lists to store selected sequences
    selected_sequences = []

    # Iterate over selected subjects and select S2 sequences per subject with replacement
    for subject in selected_subjects:
        subject_sequences = [i for i, label in enumerate(labels) if label == subject]
        selected_sequences.extend(random.choices(subject_sequences, k=S2))

    # Select the corresponding features, labels, cams, and time_seqs based on selected sequences
    selected_features = embeddings[selected_sequences]
    selected_labels = [unique_subjects.index(labels[i]) for i in selected_sequences]
    embeddings, labels = torch.as_tensor(selected_features), torch.as_tensor(selected_labels)

    open_set_loss = OpenSetLoss().cuda()(embeddings.cuda(), labels.cuda())

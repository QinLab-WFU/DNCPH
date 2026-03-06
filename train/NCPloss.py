import torch
import torch.nn.functional as F
from torch import nn


def calc_pnca_v1(embeddings, labels, proxies, scale=1):
    """
    NOTE: not support multi-hot labels
    """
    P = F.normalize(proxies)
    X = F.normalize(embeddings)
    D = torch.cdist(X, P) ** 2
    exp_neg_dists = torch.exp(-D * scale)
    denominator = ((1 - labels) * exp_neg_dists).sum(dim=1, keepdim=True)
    # no sum before mean, equals forward_paper when using one-hot labels
    losses = -torch.log(exp_neg_dists / denominator)[labels.bool()]
    return losses.mean()


def calc_pnca_v2(embeddings, labels, proxies, scale=1):
    """
    NOTE: support multi-hot labels
    """
    P = F.normalize(proxies)
    X = F.normalize(embeddings)
    D = torch.cdist(X, P) ** 2  # Euclidean squared distance @ 3.1 of ProxyNCA++
    # only negative included in the denominator
    exp_neg_dists = torch.exp(-D * scale)
    denominator = ((1 - labels) * exp_neg_dists).sum(dim=1, keepdim=True)
    loss = torch.sum(-labels * torch.log(exp_neg_dists / denominator), dim=1)
    return loss.mean()


def calc_pnca_pp_v1(embeddings, labels, proxies, scale=1):
    """
    softmax -> log -> sum
    """
    P = F.normalize(proxies)
    X = F.normalize(embeddings)
    D = torch.cdist(X, P) ** 2
    # note that compared to proxy nca, positive included in denominator
    loss = torch.sum(-labels * F.log_softmax(-D * scale, -1), -1).mean()
    return loss


def calc_pnca_pp_v2(embeddings, labels, proxies, scale=1):
    """
    adapted from HAPPIER
    simpler
    """
    P = F.normalize(proxies)
    X = F.normalize(embeddings)
    prediction_logits = F.linear(X, P) * 2 - 2
    loss = nn.CrossEntropyLoss()(prediction_logits * scale, labels)
    return loss


def calc_pnca_pp_v3(embeddings, labels, proxies, scale=1):
    """
    softmax -> sum -> log: like NCA!
    good for multi-hot, same for one-hot
    adapted from Hyp_ViT's NCALoss
    """
    P = F.normalize(proxies)
    X = F.normalize(embeddings)
    # B x K, C x K -> B x C
    D = torch.cdist(X, P) ** 2
    # D.fill_diagonal_(torch.finfo(D.dtype).max) # <- must remove this
    exp = F.softmax(-D * scale, dim=1)
    exp = torch.sum(exp * labels, dim=1)  # <- like NCA!
    non_zero = exp != 0
    loss = -torch.log(exp[non_zero]).mean()
    return loss


class NCPLoss(nn.Module):
    """
    all support one-hot samples,
    when using multi-hot:
    labels:   [0,   1,   0,   0,   1]
    e^(-d):   [0.4, 0.7, 0.3, 0.5, 0.9]
    ---
    *_pnca_v1:  -1*(log[0.7/1.2], log[0.9/1.2], ...).mean() -> ProxyNCA, not support
    *_pnca_v2:  -1*(log[0.7/1.2] + log[0.9/1.2], ...).mean() -> ProxyNCA
    *_pp_v1/v2: -1*(log[0.7/2.8] + log[0.9/2.8], ...).mean() -> ProxyNCA++
    *_pp_v3:    -1*(log[(0.7+0.9)/2.8], ...).mean() -> ProxyNCA++, more like NCA

    adapted from:
      https://github.com/dichotomies/proxy-nca
    """

    def __init__(self, nb_classes, sz_embedding, smoothing_const=0.1, scaling_x=1, scaling_p=1):
        super().__init__()
        self.nb_classes = nb_classes
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p
        self.log_first = False

        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        self.proxies = nn.Parameter(torch.randn(nb_classes, sz_embedding) / 8)

    def forward(self, X, T):
        P = F.normalize(self.proxies, p=2, dim=-1) * self.scaling_p
        X = F.normalize(X, p=2, dim=-1) * self.scaling_x
        D = torch.cdist(X, P) ** 2
        # Optional: BNInception uses label smoothing, apply it for retraining also
        # "Rethinking the Inception Architecture for Computer Vision", p. 6
        T = T * (1 - self.smoothing_const)
        T[T == 0] = self.smoothing_const / (self.nb_classes - 1)
        if self.log_first:
            # note that compared to proxy nca, positive included in denominator
            loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        else:
            # my mod
            exp = F.softmax(-D, dim=-1)
            exp = torch.sum(exp * T, dim=-1)  # <- like NCA!
            non_zero = exp != 0
            loss = -torch.log(exp[non_zero])
        return loss.mean()


if __name__ == "__main__":
    from _utils import gen_test_data

    B = 128
    C = 10
    K = 32
    scale = 10

    proxies = torch.randn((C, K))

    for is_multi_hot in [False, True]:
        embeddings, _, onehots = gen_test_data(B, C, K, is_multi_hot)
        for func in [calc_pnca_v1, calc_pnca_v2, calc_pnca_pp_v1, calc_pnca_pp_v2, calc_pnca_pp_v3]:
            loss = func(embeddings, onehots, proxies, scale)
            print(f"is_multi_hot:{is_multi_hot}", func.__name__, loss)
        print("-" * 10)

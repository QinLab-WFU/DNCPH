import datetime
import glob
import os
import pickle
import re
import time
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from loguru import logger
from openpyxl import Workbook  # pip install openpyxl
from openpyxl.reader.excel import load_workbook
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from _data import get_topk, build_loader, get_class_num


def print_in_md(rst):
    """
    print training result in Markdown format, like below:
    | K\D |  cifar   | nuswide  |  flickr  |   coco   |
    |----:|:--------:|:--------:|:--------:|:--------:|
    |  16 |    -     | 0.222@09 | 0.333@99 | 0.444@99 |
    |  32 | 0.111@99 | 0.222@99 | 0.333@09 |    -     |
    |  64 | 0.111@99 | 0.222@99 | 0.333@99 | 0.444@99 |
    | 128 |    -     |    -     | 0.333@99 | 0.444@99 |
    """
    if len(rst) == 0:
        return
    datasets = sorted(set([x["dataset"] for x in rst]), key=lambda x: get_class_num(x))
    hash_bits = sorted(set([x["hash_bit"] for x in rst]))
    best_epochs = sorted(set([x["best_epoch"] for x in rst]))

    max_len_e = len(str(best_epochs[-1]))
    max_len_k = max(len(str(hash_bits[-1])), 3)  # len("K\D") -> 3

    head = f"| {'K'+chr(92)+'D':^{max_len_k}} |"
    sept = f"|{':':->{max_len_k+2}}|"

    for dataset in datasets:
        head += f"{dataset:^{max_len_e+8}}|"  # len(" 0.xxx@ ") -> 8
        sept += f":{'-':-^{max_len_e+6}}:|"

    rows = []
    for hash_bit in hash_bits:
        temp_str = f"| {hash_bit:>{max_len_k}} |"
        for dataset in datasets:
            ret = [x for x in rst if x["dataset"] == dataset and x["hash_bit"] == hash_bit]
            if len(ret) == 1:
                temp_str += f" {ret[0]['best_map']:.3f}@{ret[0]['best_epoch']:0>{max_len_e}} |"
            else:
                temp_str += f"{'-':^{max_len_e+8}}|"
        rows.append(temp_str)

    print(head)
    print(sept)
    for row in rows:
        print(row)


def gen_random_one_hot_samples(B, C, K):
    embeddings = torch.randn(B, K).cuda()
    targets = torch.randint(low=0, high=C, size=[B]).cuda()
    labels = F.one_hot(targets, C).float()
    return embeddings, targets, labels


def gen_random_multi_hot_samples(B, C, K):
    embeddings = torch.randn(B, K).cuda()
    labels = (torch.randn(B, C) > 0.8).float().cuda()
    return embeddings, labels


def get_attr(obj, pos):
    if isinstance(pos, tuple):
        return getattr(obj, pos[0])[pos[1]]
    return getattr(obj, pos)


def set_attr(obj, pos, val):
    if isinstance(pos, tuple):
        getattr(obj, pos[0])[pos[1]] = val
    else:
        setattr(obj, pos, val)


def build_default_model(backbone, n_bits, pretrained, need_pos=False):
    if backbone == "alexnet":
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.alexnet(weights=weights)
        net.classifier[-1] = nn.Linear(4096, n_bits)
        last_layer_pos = ("classifier", -1)
    elif backbone == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, n_bits)
        last_layer_pos = "fc"
    elif backbone == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, n_bits)
        last_layer_pos = "fc"
    elif backbone == "swin_t":
        weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.swin_t(weights=weights)
        net.head = nn.Linear(net.head.in_features, n_bits)
        last_layer_pos = "head"
    elif backbone == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.vit_b_16(weights=weights)
        net.heads[-1] = nn.Linear(net.heads[-1].in_features, n_bits)
        last_layer_pos = ("heads", -1)
    else:
        raise NotImplementedError(f"not support: {backbone}")

    if pretrained:
        last_layer = get_attr(net, last_layer_pos)
        nn.init.kaiming_normal_(last_layer.weight, mode="fan_out")
        nn.init.constant_(last_layer.bias, 0)

    return (net, last_layer_pos) if need_pos else net


def init(gpu_id: str):
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # before using torch
    torch.cuda.set_device(f"cuda:{gpu_id}")
    assert torch.cuda.is_available(), "CUDA is not available"
    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_diff_same(t1, t2, dim):
    """
    Args:
        t1 = torch.tensor([1, 9, 12, 5, 24])
        t2 = torch.tensor([1, 24])
    Returns:
        diff: torch.tensor([5, 9, 12])
        same: torch.tensor([1, 24])
    From:
        https://stackoverflow.com/questions/55110047
    """
    t1 = torch.unique(t1, dim=dim)
    t2 = torch.unique(t2, dim=dim)
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(dim=dim, return_counts=True)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]
    return difference, intersection


def calc_learnable_params(*args):
    n_parameters = 0
    for x in args:
        n_parameters += sum(p.numel() for p in x.parameters() if p.requires_grad)
    return n_parameters


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def hash_center_type(n_classes, n_bits):
    """
    used in CenterHashing, CSQ, ...
    """
    lg2 = 0 if n_bits < 1 else int(math.log(n_bits, 2))
    if 2**lg2 != n_bits:
        return "random"

    if n_classes <= n_bits:
        return "ha_d"
    elif n_classes > n_bits and n_classes <= 2 * n_bits:
        return "ha_2d"
    else:
        return "random"

def gen_test_data(B, C, K, is_multi_hot=False, normalize_embeddings=True, device="cpu"):
    """
    Args:
        B: batch size
        C: number of classes
        K: dim of embeddings
        is_multi_hot: is multi-label dataset or not
        normalize_embeddings: normalize embeddings or not
        device: cpu or gpu
    Returns:
        embeddings: [B, K]
        singles: [B, ],  categorical ids, None if is_multi_hot
        onehots: [B, C], onehot encoded categorical ids
    """
    embeddings = torch.randn(B, K, requires_grad=True).to(device)
    if normalize_embeddings:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    if is_multi_hot:
        singles = None
        onehots = (torch.randn(B, C) > 0.8).float().to(device)
    else:
        singles = torch.randint(low=0, high=C, size=[B]).to(device)  # categorical id
        onehots = F.one_hot(singles, C).float().to(device)
    return embeddings, singles, onehots

#原来的
def prediction(net, dataloader, out_idx=None, use_sign=True, verbose=True):
    device = next(net.parameters()).device
    codes, clses = [], []
    net.eval()
    # logger.info(f"predicting({len(dataloader.dataset)})...")
    if verbose:
        query_iter = tqdm(dataloader, desc=f"extracting {dataloader.dataset.usage} features")
    else:
        query_iter = dataloader
    # for img, cls, _ in tqdm(dataloader):
    for x in query_iter:
        with torch.no_grad():
            out = net(x[0].to(device))
        codes.append(out if out_idx is None else out[out_idx])
        clses.append(x[1])
    return torch.cat(codes).sign() if use_sign else torch.cat(codes), torch.cat(clses).to(device)

#修改适用centerhash_vit
def prediction1(net, dataloader, out_idx=None, use_sign=True, verbose=True):
    device = next(net.parameters()).device
    codes, clses = [], []
    net.eval()
    if verbose:
        query_iter = tqdm(dataloader, desc=f"extracting {dataloader.dataset.usage} features")
    else:
        query_iter = dataloader

    for x in query_iter:
        with torch.no_grad():
            # 1. 前向传播获取模型输出（可能是元组/单张量）
            out = net(x[0].to(device))

            # 2. 关键修正：若输出是元组，提取第一个张量；否则保留原张量
            if isinstance(out, tuple):
                out = out[0]  # 取u1（或out[1]，两者值相同）

            # 3. 兼容原out_idx逻辑（若有需要，此处out已是单张量，out_idx无效可忽略）
            out = out if out_idx is None else out[out_idx]

        # 4. 收集特征和标签
        codes.append(out)
        clses.append(x[1])

    # 5. 拼接并返回（sign处理哈希编码）
    codes_cat = torch.cat(codes).sign() if use_sign else torch.cat(codes)
    clses_cat = torch.cat(clses).to(device)
    return codes_cat, clses_cat


def mean_average_precision(qB, rB, qL, rL, topk=None) -> float:
    """
    Calculate mean average precision(map).

    Args:
        qB (torch.Tensor): Query data hash code.
        rB (torch.Tensor): Database data hash code.
        qL (torch.Tensor): Query data targets, one-hot
        rL (torch.Tensor): Database data targets, one-hot
        topk (Any): Calculate top k data mAP.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = qL.shape[0]
    if topk is None:
        topk = rL.shape[0]
    mean_AP = 0.0
    for i in range(num_query):
        # Retrieve images from database
        if len(qL.shape) == 1:
            retrieval = (rL == qL[i]).float()
        elif len(qL.shape) == 2:
            retrieval = (qL[i, :] @ rL.T > 0).float()
        else:
            raise NotImplementedError(f"not support: {qL.shape}")
        # Calculate hamming distance
        hamming_dist = 0.5 * (rB.shape[1] - qB[i, :] @ rB.T)
        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]
        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()
        # Can not retrieve images
        if retrieval_cnt == 0:
            continue
        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(retrieval.device)
        # Acquire index
        index = ((retrieval == 1).nonzero(as_tuple=False).squeeze() + 1.0).float()
        mean_AP += (score / index).mean()
    mean_AP = mean_AP / num_query
    return mean_AP


def calc_hamming_dist(B1, B2):
    """
    calc Hamming distance
    Args:
        B1 (torch.Tensor): each bit of B1 ∈ {-1, 1}^k
        B2 (torch.Tensor): each bit of B2 ∈ {-1, 1}^k
    Returns:
        Hamming distance ∈ {0, k}
    """
    k = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (k - B1.mm(B2.t()))
    return distH


def pr_curve(qB, rB, query_label, retrieval_label):
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R


def p_topK(qB, rB, qL, rL, K=None):
    if K is None:
        K = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    num_query = qL.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (qL[iter].unsqueeze(0).mm(rL.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], rL.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


def cos(A, B=None):
    """cosine"""
    # An = normalize(A, norm='l2', axis=1)
    An = A / np.linalg.norm(A, ord=2, axis=1)[:, np.newaxis]
    if (B is None) or (B is A):
        return np.dot(An, An.T)
    # Bn = normalize(B, norm='l2', axis=1)
    Bn = B / np.linalg.norm(B, ord=2, axis=1)[:, np.newaxis]
    return np.dot(An, Bn.T)


def hamming(A, B=None):
    """A, B: [None, bit]
    elements in {-1, 1}
    """
    if B is None:
        B = A
    bit = A.shape[1]
    return (bit - A.dot(B.T)) // 2


def euclidean(A, B=None, sqrt=False):
    aTb = np.dot(A, B.T)
    if (B is None) or (B is A):
        aTa = np.diag(aTb)
        bTb = aTa
    else:
        aTa = np.diag(np.dot(A, A.T))
        bTb = np.diag(np.dot(B, B.T))
    D = aTa[:, np.newaxis] - 2.0 * aTb + bTb[np.newaxis, :]
    if sqrt:
        D = np.sqrt(D)
    return D


def NDCG(qF, rF, qL, rL, what=0, k=-1):
    """Normalized Discounted Cumulative Gain
    ref: https://github.com/kunhe/TALR/blob/master/%2Beval/NDCG.m
    """
    n_query = qF.shape[0]
    if (k < 0) or (k > rF.shape[0]):
        k = rF.shape[0]
    Rel = np.dot(qL, rL.T).astype(int)
    G = 2**Rel - 1
    D = np.log2(2 + np.arange(k))
    if what == 0:
        Rank = np.argsort(1 - cos(qF, rF))
    elif what == 1:
        Rank = np.argsort(hamming(qF, rF))
    elif what == 2:
        Rank = np.argsort(euclidean(qF, rF))

    _NDCG = 0
    for g, rnk in zip(G, Rank):
        dcg_best = (np.sort(g)[::-1][:k] / D).sum()
        if dcg_best > 0:
            dcg = (g[rnk[:k]] / D).sum()
            _NDCG += dcg / dcg_best
    return _NDCG / n_query


def get_precision_recall_by_Hamming_Radius(database_output, database_labels, query_output, query_labels, radius=2):
    bit_n = query_output.shape[1]
    ips = np.dot(query_output, database_output.T)
    ips = (bit_n - ips) / 2

    precX = []
    for i in range(ips.shape[0]):
        label = query_labels[i, :]
        label[label == 0] = -1
        idx = np.reshape(np.argwhere(ips[i, :] <= radius), (-1))
        all_num = len(idx)
        if all_num != 0:
            imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0
            match_num = np.sum(imatch)
            precX.append(match_num / all_num)
        else:
            precX.append(0.0)
    return np.mean(np.array(precX))


def rename_output(args):
    if os.path.exists(f"./output/{args.backbone}"):
        m_vals = [int(x.split("-m")[1]) for x in os.listdir("./output") if len(x.split("-m")) == 2]
        m_next = (max(m_vals) if len(m_vals) > 0 else 0) + 1
        print(f"rename previous output {args.backbone} -> {args.backbone}-m{m_next}")
        os.rename(f"./output/{args.backbone}", f"./output/{args.backbone}-m{m_next}")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyEval(object):
    main_dir, proj_name, backbone, dataset, n_bits = None, None, None, None, None

    def __init__(
        self,
        get_config_func,
        build_model_func,
        build_trans_func,
        build_loader_func,
        get_class_num_func,
        get_topk_func,
        proj_order,
        data_order,
        bits_order,
        the_suffix="",
    ):
        # functions for Supervised or ZeroShot Learning
        self.get_config = get_config_func
        self.build_model = build_model_func
        self.build_trans = build_trans_func
        self.build_loader = build_loader_func
        self.get_class_num = get_class_num_func
        self.get_topk = get_topk_func
        # Variables
        self.proj_order = proj_order
        self.data_order = data_order
        self.bits_order = bits_order
        self.the_suffix = the_suffix

    def prepare_excel(self, file_path, sheet_name):
        """
        Prepare the excel file to be written, create it if it doesn't exist.
        """
        is_new = False
        if os.path.exists(file_path):
            wb = load_workbook(file_path)
        else:
            wb = Workbook()

        if sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
        else:
            is_new = True
            if "Sheet" in wb.sheetnames:
                ws = wb["Sheet"]
                ws.title = sheet_name
            else:
                ws = wb.create_sheet(title=sheet_name)
        return wb, ws, is_new

    def get_ds_name(self, idx=-1):
        """
        Get the name of the database according to the different purposes.
        For example:
        1)self.dataset = "cifar"
          idx =  *: "cifar"
        2)self.dataset = (nus, voc)
          idx = -1: "nus->voc"
          idx =  0: "nus"
          idx =  1: "voc"
        """
        if isinstance(self.dataset, str):
            return self.dataset
        else:
            if idx == -1:
                return f"{self.dataset[0]}->{self.dataset[1]}"
            return self.dataset[idx]

    def write_excel_map(self, file_path, v):
        """
        support: NDCG, mAP, TrainingTime & EncodingTime
        """
        wb, ws, is_new = self.prepare_excel(file_path, self.proj_name)
        if is_new:
            for i, x in enumerate(self.bits_order["mAP"], 2):
                ws.cell(row=1, column=i).value = f"{x}bits"
            for i, x in enumerate(self.data_order, 2):
                ws.cell(row=i, column=1).value = x

        row1 = [col[0].value for col in ws.iter_cols(min_row=1, max_row=1)]
        col1 = [row[0].value for row in ws.iter_rows(min_col=1, max_col=1)]

        j = row1.index(f"{self.n_bits}bits") + 1
        i = col1.index(self.get_ds_name()) + 1

        ws.cell(row=i, column=j).value = v
        wb.save(file_path)

    def write_excel_hamming2(self, file_path, v):
        """
        save P@H≤2 result to excel.
        """
        wb, ws, is_new = self.prepare_excel(file_path, self.get_ds_name())
        if is_new:
            for i in range(len(self.proj_order)):
                ws.cell(row=1, column=i + 2).value = self.proj_order[i]
            for i, x in enumerate(self.bits_order["P@H≤2"], 2):
                ws.cell(row=i, column=1).value = f"{x}bits"

        row1 = [col[0].value for col in ws.iter_cols(min_row=1, max_row=1)]
        col1 = [row[0].value for row in ws.iter_rows(min_col=1, max_col=1)]

        j = row1.index(self.proj_name) + 1
        i = col1.index(f"{self.n_bits}bits") + 1

        ws.cell(row=i, column=j).value = v
        wb.save(file_path)

    def write_excel_pr(self, file_path, P, R):
        """
        save PR-curve result to excel.
        """
        wb, ws, is_new = self.prepare_excel(file_path, f"{self.get_ds_name()}@{self.n_bits}")
        if is_new:
            for i, x in enumerate(self.proj_order):
                ws.cell(row=1, column=2 * i + 1).value = x
                ws.cell(row=2, column=2 * i + 1).value = "R"
                ws.cell(row=2, column=2 * i + 2).value = "P"

        row1 = [col[0].value for col in ws.iter_cols(min_row=1, max_row=1)]
        try:
            j = row1.index(self.proj_name) + 1
        except ValueError:
            j = len(row1) + 1
            ws.cell(row=1, column=j).value = self.proj_name
            ws.cell(row=2, column=j).value = "R"
            ws.cell(row=2, column=j + 1).value = "P"

        for i, x in enumerate(R):
            ws.cell(row=i + 3, column=j).value = x.item()
            ws.cell(row=i + 3, column=j + 1).value = P[i].item()

        wb.save(file_path)

    def write_excel_topk(self, file_path, rst):
        """
        save TopN-precision result to excel.
        """
        wb, ws, is_new = self.prepare_excel(file_path, f"{self.get_ds_name()}@{self.n_bits}")

        j = 1
        if is_new:
            for i in range(len(self.proj_order)):
                ws.cell(row=1, column=i + 1).value = self.proj_order[i]
        else:
            while True:
                if ws.cell(row=1, column=j).value is None:
                    ws.cell(row=1, column=j).value = self.proj_name
                    break
                if ws.cell(row=1, column=j).value != self.proj_name:
                    j += 1
                    continue
                break

        for i in range(len(rst)):
            ws.cell(row=i + 2, column=j).value = rst[i].item()

        wb.save(file_path)

    def gen_cache_path(self):
        """
        Generate file path of the cache.
        """
        if isinstance(self.dataset, str):
            cache_path = f"{self.main_dir}/{self.proj_name}/output/{self.backbone}/{self.dataset}/{self.n_bits}/cache.p"
        else:
            if self.backbone is None:
                cache_path = (
                    f"{self.main_dir}/{self.proj_name}/output/{self.dataset[0]}_{self.dataset[1]}/{self.n_bits}/cache.p"
                )
            else:
                cache_path = f"{self.main_dir}/{self.proj_name}/output/{self.backbone}/{self.dataset[0]}/{self.n_bits}/{self.dataset[1]}_cache.p"
        return cache_path

    def get_checkpoint_path(self):
        """
        Get the file path of the checkpoint.
        """
        pkl_dir = os.path.dirname(self.gen_cache_path())
        pkl_list = glob.glob(f"{pkl_dir}/*.pkl")
        if len(pkl_list) != 1:
            logger.error(pkl_list)
            raise Exception(f"cannot locate one *.pkl in: {pkl_dir}")
        return pkl_list[0]

    def load_model(self):
        """
        Load pre-trained models based on configurations.
        """
        args = self.get_config()
        if isinstance(args, dict):
            args["backbone"] = self.backbone
            args["n_bits"] = self.n_bits
            if "n_classes" in args.keys():
                args["n_classes"] = self.get_class_num(self.get_ds_name(0))
        else:
            args.backbone = self.backbone
            args.n_bits = self.n_bits
            if "n_classes" in args:
                args.n_classes = self.get_class_num(self.get_ds_name(0))
        # build models in cuda
        out = self.build_model(args, pretrained=False)
        if isinstance(out, tuple):
            net, out_idx = out
        else:
            net = out
            out_idx = None
        # load checkpoint
        checkpoint_path = self.get_checkpoint_path()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        msg = net.load_state_dict(checkpoint, strict=False)
        logger.info(f"models loaded: {msg}")

        return net, out_idx

    def build_loaders(self):
        """
        Build query & retrieval data loaders based on configurations.
        """
        args = self.get_config()
        dataset = self.get_ds_name(1)
        data_dir = "/root/autodl-tmp/Dataset" + self.the_suffix
        if isinstance(args, dict):
            args["dataset"] = dataset
            args["data_dir"] = data_dir
        else:
            args.dataset = dataset
            args.data_dir = data_dir

        if self.build_trans is not None:
            out = self.build_trans(args)
            if isinstance(out, tuple):
                trans = out[1]
            else:
                trans = out
        else:
            logger.debug("use default transforms")
            trans = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        query_loader = self.build_loader(data_dir, dataset, "query", trans, batch_size=100, num_workers=4)
        dbase_loader = self.build_loader(data_dir, dataset, "dbase", trans, batch_size=100, num_workers=4)

        return query_loader, dbase_loader

    def load_prediction(self):
        """
        Load these four values, with cache use cache, without cache recalculate.
        """
        cache_path = self.gen_cache_path()

        if not os.path.exists(cache_path):
            qB, qL, rB, rL = self.calc_prediction()
            save_obj = {
                "qB": qB.cpu(),
                "qL": qL.cpu(),
                "rB": rB.cpu(),
                "rL": rL.cpu(),
            }
            with open(cache_path, "ab") as f:
                pickle.dump(save_obj, f)
        else:
            logger.debug("load qB, qL, rB, rL from cache")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            qB, qL, rB, rL = data["qB"].cuda(), data["qL"].cuda(), data["rB"].cuda(), data["rL"].cuda()
        return qB, qL, rB, rL

    def calc_prediction(self):
        """
        A pre-trained models is constructed to compute these 4 values.
        """
        net, out_idx = self.load_model()

        global query_loader
        global dbase_loader

        if "query_loader" in globals():
            if query_loader.dataset.name != self.get_ds_name(1):
                logger.debug(f"{query_loader.dataset.name} != {self.get_ds_name(1)}, loader need to be rebuilt")
                del query_loader
                del dbase_loader
            else:
                logger.debug(f"loader is the same, no need to be rebuilt")

        if "query_loader" not in globals():
            query_loader, dbase_loader = self.build_loaders()

        qB, qL = prediction(net, query_loader, out_idx)
        rB, rL = prediction(net, dbase_loader, out_idx)

        return qB, qL, rB, rL

    def get_training_time(self):
        """
        Get the training time of the best map through the log file.
        """
        log_path = os.path.dirname(self.gen_cache_path()) + "/train.log"

        with open(log_path, "r") as f:
            lines = f.read().splitlines()
        ret = []
        epoch = 0
        for line in lines:
            m = re.search(r".+\[Training].+\[epoch:([0-9./]+)]\[time:([0-9.]+)].+", line)
            if m:
                ret.append(m.group(2))
                epoch = int(m.group(1).split("/")[0])
                # print(epoch, m.group(2))
        ret = np.array(ret).astype(float)
        assert epoch != 0, f"can't find training time"
        assert len(ret) == epoch + 1, f"may missing some training time"

        idx = 0
        for line in lines[-5:]:
            # print(line)
            m = re.search(r".+best epoch: (\d+).+", line)
            if m:
                idx = int(m.group(1))
                # print(m.group(1))
                break
        assert idx != 0, "can't find best epoch"

        return ret[: idx + 1].sum()

    def get_encoding_time(self):
        """
        Get the encoding time of the models-generated hash based on the following parameters:
        dataset = "voc"
        batch_size = 100
        usage = "query"
        """
        trans = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        dataloader = self.build_loader(
            "../_datasets_zs", "voc", "query", trans, batch_size=100, shuffle=False, num_workers=4
        )
        net, _ = self.load_model()
        net.eval()
        logger.info(f"predicting({len(dataloader.dataset)})...")
        tic = time.time()
        for x in dataloader:
            with torch.no_grad():
                net(x[0].cuda())
        toc = time.time()
        return toc - tic

    def get_run_list(self, evals, n_bits):
        """
        Check whether to run an evaluation based on the number of bits.
        """
        rst = []
        for x in evals:
            if n_bits in self.bits_order[x]:
                rst.append(x)
        return rst

    def __call__(
        self,
        main_dir,
        proj_name,
        backbone,
        evals,
        datasets,
        hash_bits,
        use_cache=True,
        save_xlsx=True,
    ):
        self.main_dir = main_dir
        self.proj_name = proj_name
        self.backbone = backbone
        for dataset in datasets:
            self.dataset = dataset
            logger.info(f"processing dataset: {self.get_ds_name()}")

            for hash_bit in hash_bits:
                self.n_bits = hash_bit
                logger.info(f"processing hash-bit: {hash_bit}")

                run_list = self.get_run_list(evals, hash_bit)

                if len(run_list) == 0:
                    logger.info(f"no eval to run, pass")
                    continue

                if self.get_ds_name(0) == self.get_ds_name(1) or proj_name in ["G_MLZSL", "T_MLZSH"]:
                    is_useful = True
                else:
                    is_useful = False

                if not (
                    (
                        not is_useful
                        and set(run_list) <= {"TrainingTime", "EncodingTime", "SelfCheck"}
                        or (is_useful and set(run_list) <= {"TrainingTime", "EncodingTime"})
                    )
                ):
                    if use_cache:
                        qB, qL, rB, rL = self.load_prediction()
                    else:
                        qB, qL, rB, rL = self.calc_prediction()

                # calc mAP
                if "mAP" in run_list:
                    topk = self.get_topk(self.get_ds_name(1))
                    map_k = mean_average_precision(qB, rB, qL, rL, topk)
                    if save_xlsx:
                        self.write_excel_map(f"{main_dir}/eval_map{self.the_suffix}.xlsx", f"{map_k:.3f}")
                    logger.info(f"[dataset:{self.get_ds_name()}][bits:{hash_bit}][mAP@{topk}:{map_k:.3f}]")
                else:
                    map_k = None

                # calc PR curve
                if "PR-curve" in run_list:
                    P, R = pr_curve(qB, rB, qL, rL)
                    if save_xlsx:
                        self.write_excel_pr(f"{main_dir}/eval_pr{self.the_suffix}.xlsx", P, R)
                    logger.info(f"[dataset:{self.get_ds_name()}][bits:{hash_bit}][PR-curve is done]")

                # calc TopN precision
                if "TopN-precision" in run_list:
                    rst = p_topK(qB, rB, qL, rL)
                    if save_xlsx:
                        self.write_excel_topk(f"{main_dir}/eval_topk{self.the_suffix}.xlsx", rst)
                    logger.info(f"[dataset:{self.get_ds_name()}][bits:{hash_bit}][TopN-precision is done]")

                if "NDCG" in run_list or "P@H≤2" in run_list:
                    qB = qB.cpu().numpy()
                    rB = rB.cpu().numpy()
                    qL = qL.cpu().numpy()
                    rL = rL.cpu().numpy()

                # calc NDCG
                if "NDCG" in run_list:
                    ndcg = NDCG(qB, rB, qL, rL, what=1, k=1000)
                    if save_xlsx:
                        self.write_excel_map(f"{main_dir}/eval_ndcg{self.the_suffix}.xlsx", f"{ndcg:.3f}")
                    logger.info(f"[dataset:{self.get_ds_name()}][bits:{hash_bit}][NDCG:{ndcg:.3f}]")

                # calc Precision curves within Hamming Radius 2
                if "P@H≤2" in run_list:
                    prec = get_precision_recall_by_Hamming_Radius(rB, rL, qB, qL)
                    if save_xlsx:
                        self.write_excel_hamming2(f"{main_dir}/eval_hamming2{self.the_suffix}.xlsx", prec)
                    logger.info(f"[dataset:{self.get_ds_name()}][bits:{hash_bit}][P@H≤2:{prec:.3f}]")

                if "TrainingTime" in run_list:
                    rst = self.get_training_time()
                    if save_xlsx:
                        self.write_excel_map(f"{main_dir}/training_time{self.the_suffix}.xlsx", rst)
                    logger.info(
                        f"[dataset:{self.get_ds_name()}][bits:{hash_bit}][TrainingTime:{datetime.timedelta(seconds=rst)}]"
                    )

                if "EncodingTime" in run_list:
                    rst = self.get_encoding_time()
                    if save_xlsx:
                        self.write_excel_map(f"{main_dir}/encoding_time{self.the_suffix}.xlsx", rst)
                    logger.info(
                        f"[dataset:{self.get_ds_name()}][bits:{hash_bit}][EncodingTime:{datetime.timedelta(seconds=rst)}]"
                    )

                if "SelfCheck" in run_list:
                    # get mAP from check pint file name
                    pkl_fn = os.path.basename(self.get_checkpoint_path())
                    m = re.search(r"e\d+_([0-9.]+)\.pkl", pkl_fn)
                    if m:
                        map_pkl = float(m.group(1))
                    else:
                        m = re.search(r"iter\d+_([0-9.]+)\.pkl", pkl_fn)
                        if m:
                            map_pkl = float(m.group(1))
                        else:
                            raise Exception(f"can't extract mAP from file name: {pkl_fn}")
                    # get mAP from models prediction
                    temp = None
                    if is_useful:
                        if map_k is None:
                            topk = self.get_topk(self.get_ds_name(1))
                            map_k = mean_average_precision(qB, rB, qL, rL, topk)
                    else:
                        topk = self.get_topk(self.get_ds_name(0))
                        temp = self.dataset
                        self.dataset = (self.dataset[0], self.dataset[0])
                        # logger.debug(self.dataset)
                        if use_cache:
                            qB, qL, rB, rL = self.load_prediction()
                        else:
                            qB, qL, rB, rL = self.calc_prediction()
                        map_k = mean_average_precision(qB, rB, qL, rL, topk)

                    map_k = round(map_k.item(), 3)

                    if map_k != map_pkl:
                        # logger.warning(f"map[{map_k}] != map_pkl[{map_pkl}]")
                        logger.warning(
                            f"[dataset:{self.get_ds_name()}][bits:{hash_bit}][SelfCheck:failed][mAP@{topk}:{map_k}][mAP@pkl:{map_pkl}]"
                        )
                    else:
                        # logger.debug(f"map[{map_k}] = map_pkl[{map_pkl}]")
                        logger.info(
                            f"[dataset:{self.get_ds_name()}][bits:{hash_bit}][SelfCheck:passed][mAP@{topk}:{map_k}][mAP@pkl:{map_pkl}]"
                        )
                    if temp is not None:
                        self.dataset = temp


def init_my_eval(get_config, build_model, build_trans):
    proj_order = ["CSQ", "DPSH", "DSH",  "OrthoHash", "IDHN", "HyP2", "CenterHashing", "HashNet", "HHF", "GreedHash", "DHN", "CFBH", "DA2H"]
    data_order = ["cifar"]
    bits_order = {
        "mAP": [16, 32, 64, 128],
        "NDCG": [16, 32, 64, 128],
        "PR-curve": [16, 32,  64, 128],
        "TopN-precision": [16, 32,  64, 128],
        # "P@H≤2": [16, 32, 48, 64],
        "P@H≤2": [16, 32, 64, 128],
        "TrainingTime": [16, 32,  64, 128],
        "EncodingTime": [16, 32,  64, 128],
        "SelfCheck": [16, 32,  64, 128],
    }
    my_eval = MyEval(
        get_config, build_model, build_trans, build_loader, get_class_num, get_topk, proj_order, data_order, bits_order
    )
    return my_eval


if __name__ == "__main__":
    t1 = torch.tensor([1, 9, 9, 12, 5, 24])
    t2 = torch.tensor([1, 24])
    d, s = find_diff_same(t1, t2, 0)
    print(d)
    print(s)

import os
import json
import numpy as np
from typing import List, Dict, Any
from typing import List, Dict, Any
import random
from collections import defaultdict
from itertools import combinations


label_map = {
    "Phone Number": "Phone Number",
    "Mobile Number": "Phone Number",
    "Postal Home Address": "Location",
    "Postal Work Address": "Location",
    "Postal Address" :"Location",
    "Date of Birth": "Date",
    "Full Date": "Date",
    "Sex": "Gender",
    "Gender": "Gender",
    "Natural Person Name": "Person Name",
}


def load_dimensions_map(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    file_dimensions_map = {}

    for entry in data:
        file_name = entry.get("data", {}).get("ocr")
        if file_name:
            file_name = file_name.split("/")[-1]  # Extract just the filename (e.g., votes_fhfb0066_page1.png)

            # Get the first result object from annotations
            results = entry.get("annotations", [])[0].get("result", [])
            if results:
                first_result = results[0]
                width = first_result.get("original_width")
                height = first_result.get("original_height")

                if width is not None and height is not None:
                    file_dimensions_map[file_name] = {
                        "original_width": width,
                        "original_height": height
                    }

    return file_dimensions_map

json_file_path = "/Volumes/MyDataDrive/thesis/code-2/data/manual-label-2.json"  # Replace with actual path
dimensions_map = load_dimensions_map(json_file_path)

def estimate_accuracies_triplet_new(L: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Triplet-based accuracy estimation (Fu et al., 2020).
    L: (n_spans × m) binary vote matrix {0,1}.
    Returns: a (m,) array of accuracies in [-1,1].
    """
    n, m = L.shape
    if m < 3:
        return np.ones(m)

    # 1) map votes to ±1
    v = 2 * L - 1   # shape (n,m), now in {-1,+1}

    # 2) pairwise correlations
    r = (v.T @ v) / n  # shape (m,m), r[j,k] ≈ E[v_j v_k]

    # 3) triplet estimates
    a = np.zeros(m)
    for j in range(m):
        ests = []
        others = [o for o in range(m) if o != j]
        for k, l in combinations(others, 2):
            denom = r[k, l]
            if abs(denom) > eps:
                ests.append(np.sqrt(abs((r[j, k] * r[j, l]) / denom)))
        if ests:
            a[j] = float(np.median(ests))

    # 4) resolve global sign so sum(a) ≥ 0
    if a.sum() < 0:
        a = -a

    return a


def infer_probs_triplet_new(L: np.ndarray,
                        a: np.ndarray,
                        pi: float = 0.5,
                        eps: float = 1e-12) -> np.ndarray:
    """
    Closed-form posterior inference (Fu et al., 2020).
    L: (n_spans × m) binary vote matrix {0,1}.
    a: (m,) annotator accuracies in [-1,1].
    pi: prior P(Y=1), default 0.5 if unknown.
    Returns: (n_spans,) posterior P(Y=1 | votes).
    """
    # map to ±1
    v = 2 * L - 1

    # log-odds update
    log_prior = np.log(pi/(1 - pi) + eps)
    log_factor = np.log((1 + a + eps) / (1 - a + eps))  # shape (m,)

    # for each span i: log_odds_i = log_prior + sum_j v[i,j] * log_factor[j]
    log_odds = log_prior + v.dot(log_factor)

    # sigmoid → posterior
    return 1 / (1 + np.exp(-log_odds))



def estimate_accuracies_triplet(L: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Triplet-based accuracy estimation (Fu et al., 2020).
    L: binary matrix of shape (n_spans, n_annotators).
    Returns: array of length m with estimated accuracies in [0,1].
    """
    n, m = L.shape
    print(f"L: {L.shape}")
    # Pairwise agreement rates
    r = np.array([[ (L[:, j] * L[:, k]).mean() for k in range(m) ] for j in range(m)])
    if m < 3:
        return np.ones(m)
    # Triplet-based estimates
    a = np.ones(m)
    for j in range(m):
        estimates: List[float] = []
        others = [o for o in range(m) if o != j]
        for idx_k in range(len(others)):
            for idx_l in range(idx_k + 1, len(others)):
                k, l = others[idx_k], others[idx_l]
                denom = r[k, l] + eps
                if denom > 0:
                    estimates.append(np.sqrt((r[j, k] * r[j, l]) / denom))
        if estimates:
            print(f"estimates length: {len(estimates)}")
            a[j] = float(np.median(estimates))
    return a


def infer_probs(L: np.ndarray, a: np.ndarray, pi: float = 0.5) -> np.ndarray:
    """
    Compute posterior probability for each span given L and accuracies.
    """
    n = L.shape[0]
    posts = np.zeros(n, dtype=float)
    for i in range(n):
        S_p, S_m = pi, 1 - pi
        for j, vote in enumerate(L[i]):
            if vote:
                S_p *= (1 + a[j]) / 2
                S_m *= (1 - a[j]) / 2
        print(f"S_p : {S_p}, S_m = {S_m}, post: {S_p / (S_p + S_m)}")
        posts[i] = S_p / (S_p + S_m)
    return posts

def infer_probs_log(L: np.ndarray, a: np.ndarray, c: float = 0.3, eps: float = 1e-12, pi: float = 0.1):
    """
    Compute posterior P(span i is real) for each row i of L,
    using log-space updates for numerical stability.
    L: (n_spans × n_annotators) binary vote matrix
    a: array of length n_annotators with accuracies in [0,1]
    Returns: array of length n_spans with posteriors in (0,1)
    """
    n, m = L.shape
    posts = np.zeros(n)
    logpi   = np.log(pi + eps)
    log1mpi = np.log(1 - pi + eps)
    for i in range(n):
        log_sp = logpi
        log_sm = log1mpi
        for j, vote in enumerate(L[i]):
            if vote:
                log_sp += np.log((1 + a[j]) / 2 + eps)
                log_sm += np.log((1 - a[j]) / 2 + eps)
            else:
                log_sp += np.log((1 - a[j]) / 2 + eps)
                log_sm += np.log((1 + a[j]) / 2 + eps)
        # back to probability space
        posts[i] = 1 / (1 + np.exp(log_sm - log_sp))
    return posts


def infer_probs_log_weighted_majority_voting(L: np.ndarray, a: np.ndarray, c: float = 0.3, eps: float = 1e-12, pi: float = 0.5):
    """
    Compute posterior P(span i is real) for each row i of L,
    using log-space updates for numerical stability.
    L: (n_spans × n_annotators) binary vote matrix
    a: array of length n_annotators with accuracies in [0,1]
    Returns: array of length n_spans with posteriors in (0,1)
    """
    n, m = L.shape
    posts = np.zeros(n)
    logpi   = np.log(pi + eps)
    log1mpi = np.log(1 - pi + eps)
    for i in range(n):
        log_sp = logpi
        log_sm = log1mpi
        for j, vote in enumerate(L[i]):
            weight = 1/3
            if vote:
                log_sp += np.log((1 + weight) / 2 + eps)
                log_sm += np.log((1 - weight) / 2 + eps)
            else:
                log_sp += np.log((1 - weight) / 2 + eps)
                log_sm += np.log((1 + weight) / 2 + eps)
        # back to probability space
        posts[i] = 1 / (1 + np.exp(log_sm - log_sp))
    return posts


def score_all_jsons_global(votes_dir: str, out_dir:str):
    """
    Process all votes_*.json: group spans by PII type across all files,
    compute triplet-based accuracies & posteriors per type,
    attach probabilities, and write scored files.
    """
    # 1) Load files
    records_by_file: Dict[str, List[Dict[str, Any]]] = {}
    all_recs: List[Dict[str, Any]] = []
    order_map: List[tuple] = []  # (file_path, local_index)
    for fname in sorted(os.listdir(votes_dir)):
        if not (fname.startswith('votes_') and fname.endswith('.json')):
            continue
        path = os.path.join(votes_dir, fname)
        with open(path, 'r') as f:
            items = json.load(f)
        records_by_file[path] = items
        for idx, rec in enumerate(items):
            all_recs.append(rec)
            order_map.append((path, idx))

    if not all_recs:
        print("No vote files to process.")
        return

    # 2) Determine number of annotators
    # Get max array size from votes arrays to determine number of annotators
    m = max((len(rec.get('votes', [])) for rec in all_recs), default=0)
    print(f"m: {m}")

    # 3) Group indices by PII type (hashable key)
    type_to_indices: Dict[Any, List[int]] = defaultdict(list)
    for idx, rec in enumerate(all_recs):
        key = rec.get('pii_type')
        # normalize list keys to tuple
        if isinstance(key, list):
            key = tuple(key)
        type_to_indices[key].append(idx)

    # 4) Compute probabilities per type
    probs = np.zeros(len(all_recs), dtype=float)
    for pii_type, idxs in type_to_indices.items():
        # Build label matrix for this type
        L = np.zeros((len(idxs), m), dtype=int)
        for row_i, rec_idx in enumerate(idxs):
            votes = all_recs[rec_idx].get('votes', [])
            for j, vote in enumerate(votes):
                L[row_i, j] = vote
        # Estimate accuracies & infer posteriors
        a = estimate_accuracies_triplet_new(L)
        print(f"Annotators Accuracy : {a} for PII type : {pii_type}")
        posts = infer_probs_triplet_new(L, a)
        for i, rec_idx in enumerate(idxs):
            probs[rec_idx] = posts[i]

    # 5) Attach probabilities back in each file's list
    for (path, local_idx), p in zip(order_map, probs.tolist()):
        records_by_file[path][local_idx]['probability'] = float(p)
        file_name = path.split("/")[-1].split(".")[0] + ".png"
        bb = dimensions_map[file_name]

        # Here we change to abosolute bbox
        x = records_by_file[path][local_idx]['bbox'][0] * bb['original_width']
        y = records_by_file[path][local_idx]['bbox'][1] * bb['original_height']
        width =  records_by_file[path][local_idx]['bbox'][2] * bb['original_width']
        height = records_by_file[path][local_idx]['bbox'][3] * bb['original_height']

        x0 = round(x)
        y0 = round(y)
        x1 = round(width)
        y1 = round(height)

        int_bbox = [x0, y0, x1, y1]

        records_by_file[path][local_idx]['bbox'] = int_bbox


        if isinstance(records_by_file[path][local_idx]['pii_type'], list):
            new_list = []
            for i in records_by_file[path][local_idx]['pii_type']:
                new_list.append(label_map.get(i, i))
            records_by_file[path][local_idx]['pii_type'] = new_list
        else:
            records_by_file[path][local_idx]['pii_type'] = label_map.get(
                records_by_file[path][local_idx]['pii_type'],
                records_by_file[path][local_idx]['pii_type']
            )

        # print(records_by_file[path][local_idx]['pii_type'])


    # 5: write back per file
    for path, items in records_by_file.items():
        # out_path = path.replace('.json', '_scored.json')
        file_name = path.split("/")[-1]
        # out_path = f"/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/test-final/{file_name}"
        out_path = out_dir + file_name
        with open(out_path, 'w') as f:
            json.dump(items, f, indent=4)
        print(f"Wrote scored file: {out_path}")

from annoy import AnnoyIndex
from typing import Callable, Optional, Iterable, Tuple, List
import numpy as np


def buildIndex(dim: int, data: Iterable[Tuple[int, list]], n_trees: int, metric: str = "manhattan",
               seed: Optional[int] = None, on_disk_build: bool = False, path: str = "__annoy_index.ai") -> AnnoyIndex:
    """builds an AnnoyIndex object from given data"""

    annoy_index = AnnoyIndex(dim, metric)
    if on_disk_build:
        annoy_index.on_disk_build(path)
    for i, vec in data:
        annoy_index.add_item(i, vec)
    if seed:
        annoy_index.set_seed(seed)
    annoy_index.build(n_trees)
    return annoy_index


def find_matches(ai: AnnoyIndex, data: Iterable[Tuple[int, list]], threshold: float = .01, threshold_avg=False,
                 avg_len=1, norm_ord: int = 1, metric: Optional[Callable] = None) -> List[Tuple[int, int]]:
    """finds pairs of matching records between 'data' (Iterable of key,value pairs) and 'ai'"""

    if metric is None:
        #metric = lambda x, y: np.linalg.norm(x - y, ord=norm_ord)
        def metric(x, y):
            intersection = np.logical_and(np.asarray(x), np.asarray(y))
            return 1 - (2. * intersection.sum() / (np.asarray(x).sum() + np.asarray(y).sum()))
    matches = []
    for i, vec in data:
        nearests = ai.get_nns_by_vector(vec, 50, include_distances=False)
        candidate = None
        cand_dist = -1
        for nearest in nearests:
            sim = 1 - metric(ai.get_item_vector(nearest), vec)
            if sim > cand_dist:
                if threshold_avg:
                    if sim >= threshold * avg_len:
                        candidate = (nearest, i)
                        cand_dist = sim
                else:
                    if sim >= 1 - threshold:
                        candidate = (nearest, i)
                        cand_dist = sim
        if candidate is not None:
            matches.append(candidate)
    return matches

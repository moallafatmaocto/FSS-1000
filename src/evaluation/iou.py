import numpy as np


def iou(mask_1: np.ndarray, mask_2: np.ndarray) -> float:
    return positive_areas_intersection(mask_1, mask_2) / positive_areas_union(mask_1, mask_2)


def positive_areas_union(mask_1, mask_2) -> float:
    positive_union_mask = mask_1 + mask_2 >= 1
    return float(np.sum(positive_union_mask))


def positive_areas_intersection(mask_1, mask_2) -> float:
    positive_intersection_mask = mask_1 + mask_2 == 2
    return float(np.sum(positive_intersection_mask))

import numpy as np

from typing import List, Tuple

# TODO: Add type hints


def iou(pred_box, target_box):

    if len(target_box.shape) == 1:
        target_box = target_box[np.newaxis, :]

    xA = np.maximum(pred_box[0], target_box[:, 0])
    yA = np.maximum(pred_box[1], target_box[:, 1])
    xB = np.minimum(pred_box[2], target_box[:, 2])
    yB = np.minimum(pred_box[3], target_box[:, 3])

    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)

    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (target_box[:, 2] - target_box[:, 0]) * \
        (target_box[:, 3] - target_box[:, 1])

    scores = intersection / (boxAArea + boxBArea - intersection)
    return scores


def nms(rect_list: List[Tuple[Tuple[int, int, int, int], float]],
        threshhold: float) -> List[Tuple[Tuple[int, int, int, int], float]]:

    nms_list: List[Tuple[Tuple[int, int, int, int], float]] = []
    rect_array = np.array([rect[0] for rect in rect_list])
    score_array = np.array([rect[1] for rect in rect_list])

    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    while len(score_array) > 0:
        nms_list.append((rect_array[0], score_array[0]))
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        length = len(score_array)
        if length <= threshhold:
            break

        iou_scores = iou(np.array(nms_list[-1][0]), rect_array)
        idxs = np.where(iou_scores < threshhold)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_list

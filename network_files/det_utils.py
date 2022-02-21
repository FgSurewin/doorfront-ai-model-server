import torch
import math
from typing import List, Tuple
from torch import Tensor

from train_utils.train_eval_utils import box_iou
from train_utils.train_eval_utils import single_box_iou


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float) -> None
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        # 遍历每张图像的matched_idxs
        for matched_idxs_per_image in matched_idxs:
            # >= 1的为正样本, nonzero返回非零元素索引
            # positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            # = 0的为负样本
            # negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            # 指定正样本的数量
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            # 如果正样本数量不够就直接采用所有正样本
            num_pos = min(positive.numel(), num_pos)
            # 指定负样本数量
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            # 如果负样本数量不够就直接采用所有负样本
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            # Returns a random permutation of integers from 0 to n - 1.
            # 随机选择指定数量的正负样本
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes(gt)
        proposals (Tensor): boxes to be encoded(anchors)
        weights:
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # unsqueeze()
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    # parse widths and heights
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    # parse coordinate of center point
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        结合anchors和与之对应的gt计算regression参数
        Args:
            reference_boxes: List[Tensor] 每个proposal/anchor对应的gt_boxes
            proposals: List[Tensor] anchors/proposals

        Returns: regression parameters

        """
        # 统计每张图像的anchors个数，方便后面拼接在一起处理后在分开
        # reference_boxes和proposal数据结构相同
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        # targets_dx, targets_dy, targets_dw, targets_dh
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        """

        Args:
            rel_codes: bbox regression parameters
            boxes: anchors/proposals

        Returns:

        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)

        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        # 将预测的bbox回归参数应用到对应anchors上得到预测bbox的坐标
        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )

        # 防止pred_boxes为空时导致reshape报错
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes (bbox regression parameters)
            boxes (Tensor): reference boxes (anchors/proposals)
        """
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]   # anchor/proposal宽度
        heights = boxes[:, 3] - boxes[:, 1]  # anchor/proposal高度
        ctr_x = boxes[:, 0] + 0.5 * widths   # anchor/proposal中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor/proposal中心y坐标

        wx, wy, ww, wh = self.weights  # RPN中为[1,1,1,1], fastrcnn中为[10,10,5,5]
        dx = rel_codes[:, 0::4] / wx   # 预测anchors/proposals的中心坐标x回归参数
        dy = rel_codes[:, 1::4] / wy   # 预测anchors/proposals的中心坐标y回归参数
        dw = rel_codes[:, 2::4] / ww   # 预测anchors/proposals的宽度回归参数
        dh = rel_codes[:, 3::4] / wh   # 预测anchors/proposals的高度回归参数

        # limit max value, prevent sending too large values into torch.exp()
        # self.bbox_xform_clip=math.log(1000. / 16)   4.135
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # xmin
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold    # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        计算anchors与每个gtboxes匹配的iou最大值，并记录索引，
        iou<low_threshold索引值为-1， low_threshold<=iou<high_threshold索引值为-2
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        # M x N 的每一列代表一个anchors与所有gt的匹配iou值
        # matched_vals代表每列的最大值，即每个anchors与所有gt匹配的最大iou值
        # matches对应最大值所在的索引
        matched_vals, matches = match_quality_matrix.max(dim=0)  # the dimension to reduce.
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        # 计算iou小于low_threshold的索引
        below_low_threshold = matched_vals < self.low_threshold
        # 计算iou在low_threshold与high_threshold之间的索引值
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        # iou小于low_threshold的matches索引置为-1
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1

        # iou在[low_threshold, high_threshold]之间的matches索引置为-2
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS    # -2

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        # 对于每个gt boxes寻找与其iou最大的anchor，
        # highest_quality_foreach_gt为匹配到的最大iou值
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  # the dimension to reduce.

        # Find highest quality match available, even if it is low, including ties
        # 寻找每个gt boxes与其iou最大的anchor索引，一个gt匹配到的最大iou可能有多个anchor
        # gt_pred_pairs_of_highest_quality = torch.nonzero(
        #     match_quality_matrix == highest_quality_foreach_gt[:, None]
        # )
        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None])
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        # gt_pred_pairs_of_highest_quality[:, 0]代表是对应的gt index(不需要)
        # pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        # 保留该anchor匹配gt最大iou的索引，即使iou低于设定的阈值
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    # cond = n < beta
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def logproposedregionstatus(labels):
    import numpy
    for label in labels:
        u, c = numpy.unique(label.cpu().numpy(), return_counts=True)
        with open('ProposedRegionCount.txt', 'a') as f:
            f.write("image proposed regions:\n")
            f.write(",".join(map(str, u)) + "\n")
            f.write(",".join(map(str, c)) + "\n")


def delete_multiple_element(list_object, indices):
    list_object = list_object.tolist()
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

    return torch.tensor(list_object)

def removeoutsideboxes(boxes, scores, labels):
    processed_boxes = []
    processed_scores = []
    processed_labels = []
    doors_idx = [i for i, x in enumerate(labels) if x == 1]
    knobs_idx = [i for i, x in enumerate(labels) if x == 2]
    removable_idx = []
    knob_iou_list = []
    if len(knobs_idx) > 0:
        if len(doors_idx) == 1:
            door_box = boxes[doors_idx[0]]
            for i in knobs_idx:
                knob_iou = single_box_iou(door_box, boxes[i])
                if knob_iou == 0:
                    removable_idx.append(i)
        elif len(doors_idx) > 1:
            if len(knobs_idx) == 1:
                knob_box = boxes[knobs_idx[0]]
                for i in doors_idx:
                    knob_iou_list.append(single_box_iou(boxes[i], knob_box))
                if max(knob_iou_list) > 0:
                    removable_idx.append(knobs_idx[0])
            elif len(knobs_idx) > 1:
                door_box_list = torch.stack([boxes[i] for i in doors_idx])
                knob_box_list = torch.stack([boxes[i] for i in knobs_idx])
                iou_matrix = box_iou(knob_box_list, door_box_list)
                for row_id, row in enumerate(iou_matrix):
                    if row.max() == 0:
                        removable_idx.append(knobs_idx[row_id])

        boxes = delete_multiple_element(boxes, removable_idx)
        scores = delete_multiple_element(scores, removable_idx)
        labels = delete_multiple_element(labels, removable_idx)

    assert len(boxes) == len(scores)
    assert len(boxes) == len(labels)

    return boxes, scores, labels


def drawboxesonoriginalimage(image_id, category_id, category_name, result_folder, boxes, image_shapes,
                             original_image_sizes, scores=None, labels=None, print_scores=False):
    import datetime
    import numpy as np
    import os
    import torchvision
    from PIL import Image, ImageFont, ImageDraw

    categories = [{'id': 0, 'name': 'None'}, {'id': 1, 'name': 'Door'}, {'id': 2, 'name': 'Knob'},
                  {'id': 3, 'name': 'Stairs'}, {'id': 4, 'name': 'Ramp'}]
    pred_color = {k["id"]: (3, 15, 252) for k in categories}

    gt_color = {0: (0, 0, 0), 1: (255, 0, 0),
                2: (0, 200, 255), 3: (0, 255, 0), 4: (255, 182, 193)}
    transform = torchvision.transforms.ToPILImage(mode='RGB')
    img = Image.open(image_id)
    font_size = int(img.size[0] * 20.0 / 1200)
    # img = np.array(img)
    font = ImageFont.truetype("arial.ttf", size=font_size)
    draw = ImageDraw.Draw(img)
    door_name = categories[1]["name"]
    knob_name = categories[2]["name"]
    stair_name = categories[3]["name"]
    boxes = resize_detections(boxes, image_shapes, original_image_sizes)
    if category_id == 0:
        for i, box in enumerate(boxes):
            draw.rectangle(box.tolist(), outline=pred_color[category_id],
                           width=int(img.size[0] / 500.0))
    else:
        target_idx_list = [i for i, x in enumerate(labels) if x == category_id]
        target_boxes = [boxes[i] for i in target_idx_list]
        for idx, box in enumerate(target_boxes):
            draw.rectangle(box.tolist(), outline=pred_color[category_id],
                           width=int(img.size[0] / 500.0))
            if print_scores:
                text = category_name + "_" + str(np.round(scores[target_idx_list[idx]].item(), 2))
                draw.text((box.tolist()[0] + 1, box.tolist()[1] - font_size - 1),
                          text=text, fill=gt_color[category_id], font=font)
    img.save(os.path.join(result_folder, str(image_id) + "_" + str(category_name) + ".jpg"))


def resize_detections(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """
    将boxes参数根据图像的缩放情况进行相应缩放

    Arguments:
        original_size: 图像缩放前的尺寸
        new_size: 图像缩放后的尺寸
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios
    # Removes a tensor dimension, boxes [minibatch, 4]
    # Returns a tuple of all slices along a given dimension, already without it.
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def spatial_reasoning(boxes, scores, labels):
    # find highest confidence object for each overlapped object group
    # check the spatial relation between stair and door, knob and door
    # check stair location and update confidence for both stair and door
    # check knob location inside the detected door and update confidence score
    # remove low score detections use update score threshold
    # return list of boxes, scores and labels

    original_scores = scores.tolist()
    update_scores = scores.tolist()
    overlapped_doors = overlapped_boxes(boxes, scores, labels, 1)
    overlapped_knobs = overlapped_boxes(boxes, scores, labels, 2)
    overlapped_stairs = overlapped_boxes(boxes, scores, labels, 3)
    high_conf_doors = []
    high_conf_knobs = []
    high_conf_stairs = []
    proposed_index = []
    proposed_boxes = []
    proposed_scores = []
    proposed_labels = []

    # get overlapped objects
    for doors in overlapped_doors:
        door_scores = [update_scores[i] for i in doors]
        highest_door_idx = door_scores.index(max(door_scores))
        high_conf_doors.append(doors[highest_door_idx])
        # proposed_index.append(doors[highest_door_idx])

    for knobs in overlapped_knobs:
        knob_scores = [update_scores[i] for i in knobs]
        highest_knob_idx = knob_scores.index(max(knob_scores))
        high_conf_knobs.append(knobs[highest_knob_idx])
        # proposed_index.append(knobs[highest_knob_idx])

    for stairs in overlapped_stairs:
        stair_scores = [update_scores[i] for i in stairs]
        highest_stair_idx = stair_scores.index(max(stair_scores))
        high_conf_stairs.append(stairs[highest_stair_idx])
        # proposed_index.append(stairs[highest_stair_idx])

    # get detected doors with highest confidence score
    high_conf_door_boxes = [boxes[i] for i in high_conf_doors]
    high_conf_stair_boxes = [boxes[i] for i in high_conf_stairs]
    high_conf_knob_boxes = [boxes[i] for i in high_conf_knobs]
    # proposed_boxes, proposed_scores, proposed_labels = boxes[proposed_index], scores[proposed_index], labels[proposed_index]
    # use spatial relation to update confidence score for stairs
    for door_idx, door_box in enumerate(high_conf_door_boxes):
        conf_score_door = original_scores[high_conf_doors[door_idx]]
        proposed_index.append(high_conf_doors[door_idx])
        proposed_boxes.append(door_box)
        proposed_scores.append(conf_score_door)
        proposed_labels.append(labels[high_conf_doors[door_idx]])
        if conf_score_door >= 0.8:
            stair_list = []
            knob_list = []
            for stair_idx, stair_box in enumerate(high_conf_stair_boxes):
                if has_stair_spatial_relation(door_box, stair_box):
                    stair_list.append(stair_idx)

            if len(stair_list) > 0:
                update_stair_scores = [original_scores[high_conf_stairs[i]] for i in stair_list]
                for idx in stair_list:
                    stair_obj_ids = update_highest_stair_conf_scores(idx, overlapped_stairs,
                                                                     conf_score_door, original_scores, top_k=2)
                    for stair_id in stair_obj_ids:
                        proposed_index.append(stair_id)
                        proposed_boxes.append(boxes[stair_id])
                        proposed_scores.append(original_scores[stair_id])
                        proposed_labels.append(labels[stair_id])

                    door_obj_id, update_door_score = update_highest_door_conf_scores(door_idx, overlapped_doors, original_scores,
                                                                    max(update_stair_scores), 0.87, 0.1)
                    if door_obj_id not in proposed_index:
                        proposed_index.append(door_obj_id)
                        proposed_boxes.append(boxes[door_obj_id])
                        proposed_scores.append(update_door_score)
                        proposed_labels.append(labels[door_obj_id])
                    else:
                        current_door_index = proposed_index.index(door_obj_id)
                        proposed_scores[current_door_index] = update_door_score
                    # update_scores = update_stair_conf_scores(idx, overlapped_stairs, conf_score_door, update_scores,
                    #                                          original_scores)
                    # update_scores = update_door_conf_scores(door_idx, overlapped_doors, update_scores, original_scores,
                    #                                         max(update_stair_scores), 0.87, 0.1)

            # use knob distribution to further update confidence score for knobs
            for knob_idx, knob_box in enumerate(high_conf_knob_boxes):
                if has_knob_spatial_relation(door_box, knob_box):
                    knob_list.append(knob_idx)

            if len(knob_list) > 0:
                update_knob_scores = [original_scores[high_conf_knobs[i]] for i in knob_list]
                for idx in knob_list:
                    knob_obj_ids = update_highest_knob_conf_scores(idx, overlapped_knobs, door_box, conf_score_door,
                                                                   boxes, original_scores, top_k=2)
                    for knob_id in knob_obj_ids:
                        proposed_index.append(knob_id)
                        proposed_boxes.append(boxes[knob_id])
                        proposed_scores.append(original_scores[knob_id])
                        proposed_labels.append(labels[knob_id])
                    door_obj_id, update_door_score = update_highest_door_conf_scores(door_idx, overlapped_doors, original_scores,
                                                                    max(update_knob_scores), 1, 0.3, door_weight=1)
                    if door_obj_id not in proposed_index:
                        proposed_index.append(door_obj_id)
                        proposed_boxes.append(boxes[door_obj_id])
                        proposed_scores.append(update_door_score)
                        proposed_labels.append(labels[door_obj_id])
                    else:
                        current_door_index = proposed_index.index(door_obj_id)
                        proposed_scores[current_door_index] = update_door_score
                    # update_scores = update_knob_conf_scores(idx, overlapped_knobs, door_box, conf_score_door, boxes,
                    #                                         update_scores, original_scores, )
                    # update_scores = update_door_conf_scores(door_idx, overlapped_doors, update_scores, update_scores,
                    #                                         max(update_knob_scores), 1, 0.3, door_weight=1)

    return torch.stack(proposed_boxes), torch.tensor(proposed_scores), torch.tensor(proposed_labels)
    # return boxes, torch.tensor(update_scores), labels


def has_knob_spatial_relation(door_box, knob_box):
    knob_center = [(knob_box[0] + knob_box[2]) / 2, (knob_box[1] + knob_box[3]) / 2]
    has_relation = rectContains(door_box, knob_center)

    return has_relation


def has_stair_spatial_relation(door_box, stair_box):
    # door_width = door_box[2] - door_box[0]
    door_height = door_box[3] - door_box[1]
    stair_width = stair_box[2] - stair_box[0]
    stair_height = stair_box[3] - stair_box[1]
    stair_center = [stair_box[0] + stair_width / 2, stair_box[1] + stair_height / 2]
    search_area = [door_box[0] - stair_width / 2, door_box[3] - door_height * 0.2,
                   door_box[2] + stair_width / 2, stair_box[3]]

    has_relation = rectContains(search_area, stair_center)

    return has_relation


def update_door_conf_scores(idx, overlapped_objects, scores, original_scores, related_obj_score, relation_score,
                            conditional_weight, door_weight=0.6):
    update_object_list = overlapped_objects[idx]
    for i in update_object_list:
        scores[i] = min(1, door_weight * original_scores[i] + conditional_weight * related_obj_score * relation_score)

    return scores


def update_stair_conf_scores(idx, overlapped_objects, door_score, scores, original_scores):
    stair_weight = 0.7
    conditional_weight = 0.3
    stair_over_door_score = 0.21
    update_object_list = overlapped_objects[idx]
    for i in update_object_list:
        scores[i] = min(1, stair_weight * original_scores[i] +
                        conditional_weight * door_score * stair_over_door_score)

    return scores


def update_knob_conf_scores(idx, overlapped_objects, door_box, door_score, boxes, scores, original_scores):
    knob_weight = 0.7
    conditional_weight = 0.3
    update_object_list = overlapped_objects[idx]
    for i in update_object_list:
        knob_center = [(boxes[i][0]+boxes[i][2]) / 2,
                       (boxes[i][1]+boxes[i][3]) / 2]
        knob_over_door_score = check_knob_location(door_box, knob_center)
        scores[i] = min(1, knob_weight * original_scores[i] +
                        conditional_weight * door_score * knob_over_door_score)

    return scores


def update_highest_door_conf_scores(idx, overlapped_objects, original_scores, related_obj_score, relation_score,
                                    conditional_weight, door_weight=0.6):
    update_object_id = overlapped_objects[idx][0]
    object_score = min(1, door_weight * original_scores[update_object_id] +
                       conditional_weight * related_obj_score * relation_score)

    # original_scores[update_object_id] = object_score
    return update_object_id, object_score


def update_highest_stair_conf_scores(idx, overlapped_objects, door_score, original_scores, top_k=100):
    stair_weight = 0.7
    conditional_weight = 0.3
    stair_over_door_score = 0.21
    update_object_ids = overlapped_objects[idx][:top_k]
    for update_object_id in update_object_ids:
        original_scores[update_object_id] = min(1, stair_weight * original_scores[update_object_id] +
                                                conditional_weight * door_score * stair_over_door_score)

    return update_object_ids


def update_highest_knob_conf_scores(idx, overlapped_objects, door_box, door_score, boxes, original_scores, top_k=100):
    knob_weight = 0.7
    conditional_weight = 0.3
    update_object_ids = overlapped_objects[idx][:top_k]
    for update_object_id in update_object_ids:
        knob_center = [(boxes[update_object_id][0]+boxes[update_object_id][2]) / 2,
                       (boxes[update_object_id][1]+boxes[update_object_id][3]) / 2]
        knob_over_door_score = check_knob_location(door_box, knob_center)
        original_scores[update_object_id] = min(1, knob_weight * original_scores[update_object_id] +
                                                conditional_weight * door_score * knob_over_door_score)

    return update_object_ids


def check_knob_location(door_box, knob_center):
    door_width = door_box[2] - door_box[0]
    door_height = door_box[3] - door_box[1]
    bin_score = 0
    location_score = [0, 0.01, 0,
                      0.21, 0.53, 0.19,
                      0.01, 0.05, 0.01]
    if (door_box[0] <= knob_center[0] < (door_box[0] + door_width / 3)) and (
            door_box[1] <= knob_center[1] < (door_box[1] + door_height / 3)):
        bin_score = location_score[0]
    elif (door_box[0] + door_width / 3) <= knob_center[0] < (door_box[0] + (door_width * 2) / 3) and (
            door_box[1] <= knob_center[1] < (door_box[1] + door_height / 3)):
        bin_score = location_score[1]
    elif ((door_box[0] + door_width * 2 / 3) <= knob_center[0] < door_box[2]) and (
            door_box[1] <= knob_center[1] < (door_box[1] + door_height / 3)):
        bin_score = location_score[2]
    elif (door_box[0] <= knob_center[0] < (door_box[0] + door_width / 3)) and (
            (door_box[1] + door_height / 3) <= knob_center[1] < (door_box[1] + (door_height * 2) / 3)):
        bin_score = location_score[3]
    elif (door_box[0] + door_width / 3) <= knob_center[0] < (door_box[0] + (door_width * 2) / 3) and (
            (door_box[1] + door_height / 3) <= knob_center[1] < (door_box[1] + (door_height * 2) / 3)):
        bin_score = location_score[4]
    elif ((door_box[0] + door_width * 2 / 3) <= knob_center[0] < door_box[2]) and (
            (door_box[1] + door_height / 3) <= knob_center[1] < (door_box[1] + (door_height * 2) / 3)):
        bin_score = location_score[5]
    elif (door_box[0] <= knob_center[0] < (door_box[0] + door_width / 3)) and (
            (door_box[1] + door_height * 2 / 3) <= knob_center[1] < door_box[3]):
        bin_score = location_score[6]
    elif (door_box[0] + door_width / 3) <= knob_center[0] < (door_box[0] + (door_width * 2) / 3) and (
            (door_box[1] + door_height * 2 / 3) <= knob_center[1] < door_box[3]):
        bin_score = location_score[7]
    elif ((door_box[0] + door_width * 2 / 3) <= knob_center[0] < door_box[2]) and (
            (door_box[1] + door_height * 2 / 3) <= knob_center[1] < door_box[3]):
        bin_score = location_score[8]

    return bin_score


def overlapped_boxes(boxes, scores, labels, cat_id):
    cat_idx = [i for i, x in enumerate(labels) if x == cat_id]
    original_scores = scores
    category_overlap_list = []

    if len(cat_idx) > 0:
        cat_boxes = torch.stack([boxes[i] for i in cat_idx])
        cat_iou_matrix = box_iou(cat_boxes, cat_boxes)
        for row_id, row in enumerate(cat_iou_matrix):
            if len(category_overlap_list) > 0:
                if not in_nested_list(category_overlap_list, cat_idx[row_id]):
                    overlap_boxes = []
                    for col_id, col_val in enumerate(row):
                        if col_val.item() > 0:
                            if not in_nested_list(category_overlap_list, cat_idx[col_id]):
                                overlap_boxes.append(cat_idx[col_id])
                    category_overlap_list.append(overlap_boxes)
            else:
                overlap_boxes = []
                for col_id, col_val in enumerate(row):
                    if col_val.item() > 0:
                        overlap_boxes.append(cat_idx[col_id])
                category_overlap_list.append(overlap_boxes)
                # IoU threshold for overlapped door Detections

                # IoU threshold for overlapped stair Detections
    return category_overlap_list


def in_nested_list(my_list, item):
    """
    Determines if an item is in my_list, even if nested in a lower-level list.
    """
    if item in my_list:
        return True
    else:
        return any(in_nested_list(sublist, item) for sublist in my_list if isinstance(sublist, list))


def rectContains(rect,pt):
    logic = rect[0] < pt[0] < rect[2] and rect[1] < pt[1] < rect[3]
    return logic

import math
import numpy as np
import random
import os
import sys
import time
import datetime

import torch
import torchvision

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from PIL import Image, ImageFont, ImageDraw

import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    enable_amp = True if "cuda" in device.type else False
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        # print("images: ", len(images))
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # print("targets: ", len(targets))

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=enable_amp):
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purpose
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            # 记录训练损失
            mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

            if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device):

    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    categories = [{'id': 0, 'name': 'None'}, {'id': 1, 'name': 'Door'}, {'id': 2, 'name': 'Knob'},
                  {'id': 3, 'name': 'Stairs'}, {'id': 4, 'name': 'Ramp'}]
    pred_color = {k["id"]: (3, 15, 252) for k in categories}
    gt_color = {0: (0, 0, 0), 1: (255, 0, 0),
                2: (0, 200, 255), 3: (0, 255, 0), 4: (255, 182, 193)}

    result_folder = "Detection_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(result_folder, exist_ok=False)

    tp_doors = 0
    fp_doors = 0
    tp_knobs = 0
    fp_knobs = 0
    tp_stairs = 0
    fp_stairs = 0
    tp_ramps = 0
    fp_ramps = 0
    total_doors = 0
    total_knobs = 0
    total_stairs = 0
    total_ramps = 0
    total_detected_doors = 0
    total_detected_knobs = 0
    total_detected_stairs = 0
    total_detected_ramps = 0

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        # print(outputs)        

        with open('test_result.txt', 'a') as f:
            for idx, target in enumerate(targets):
                transform = torchvision.transforms.ToPILImage(mode='RGB')
                img = image[idx].cpu()
                img = transform(img)
                img_id = target["image_id"].numpy()[0]
                f.write("Image_ID: " + str(target["image_id"].numpy()) + "\n")
                current_target = targets[idx]
                current_output = outputs[idx]
                gt_labels = current_target["labels"]
                gt_boxes = current_target["boxes"]
                predicted_labels = current_output["labels"]
                predicted_boxes = current_output["boxes"]
                predicted_scores = current_output["scores"]
                isTensor = torch.is_tensor(predicted_labels) or torch.is_tensor(gt_labels)
                predicted_boxes, predicted_scores, predicted_labels = processed_predictions(gt_boxes, gt_labels,
                                                                                            predicted_boxes,
                                                                                            predicted_scores,
                                                                                            predicted_labels)
                current_output["boxes"] = predicted_boxes
                current_output["labels"] = predicted_labels
                current_output["scores"] = predicted_scores
                if isTensor:
                    gt_labels = gt_labels.tolist()
                    gt_boxes = gt_boxes.tolist()
                    predicted_labels = predicted_labels.tolist()
                    predicted_boxes = predicted_boxes.tolist()
                    predicted_scores = predicted_scores.tolist()                
                update_idx = 0
                change = False
                if len(current_output["boxes"]) > 0:
                    # calculate success/failed detections
                    match_quality_matrix = box_iou(current_target["boxes"], current_output["boxes"])
                    for label in current_output["labels"]:
                        if label == 1:
                            total_detected_doors += 1
                        if label == 2:
                            total_detected_knobs += 1
                        if label == 3:
                            total_detected_stairs += 1
                        if label == 4:
                            total_detected_ramps += 1
                    for row_id, row in enumerate(match_quality_matrix):
                        current_gtbox = current_target["boxes"][row_id]
                        box_width = current_gtbox[2] - current_gtbox[0]
                        box_height = current_gtbox[3] - current_gtbox[1]
                        detected = False
                        pred_label_id = 0
                        for col_id, col_val in enumerate(row):
                            if current_target["labels"][row_id] == 1:
                                if col_val >= 0.5 and current_output["labels"][col_id] == 1:
                                    tp_doors += 1
                            if current_target["labels"][row_id] == 2:
                                if col_val >= 0.5 and current_output["labels"][col_id] == 2:
                                    detected = True
                                    tp_knobs += 1
                                    pred_label_id = col_id 
                            if current_target["labels"][row_id] == 3:
                                if col_val >= 0.5 and current_output["labels"][col_id] == 3:
                                    tp_stairs += 1
                            if current_target["labels"][row_id] == 4:
                                if col_val >= 0.5 and current_output["labels"][col_id] == 4:
                                    tp_ramps += 1
                        if current_target["labels"][row_id] == 2:
                            if detected:
                                f.write("Success: catid:" + str(current_target["labels"][row_id].numpy())
                                        + "," + str(current_target["boxes"][row_id].numpy()) + ", width: "
                                        + str(box_width.numpy())
                                        + ", Height: " + str(box_height.numpy()) + "\n")
                                total_knobs += 1
                            else:
                                f.write("Failed: catid:" + str(current_target["labels"][row_id].numpy())
                                        + "," + str(current_target["boxes"][row_id].numpy()) + ", width: "
                                        + str(box_width.numpy())
                                        + ", Height: " + str(box_height.numpy()) + "\n")
                                total_knobs += 1
                        else:
                            if row.max() >= 0.5:
                                f.write("Success: catid:" + str(current_target["labels"][row_id].numpy())
                                        + "," + str(current_target["boxes"][row_id].numpy()) + ", width: "
                                        + str(box_width.numpy())
                                        + ", Height: " + str(box_height.numpy()) + "\n")
                                if current_target["labels"][row_id] == 1:
                                    total_doors += 1
                                if current_target["labels"][row_id] == 3:
                                    total_stairs += 1
                                if current_target["labels"][row_id] == 4:
                                    total_ramps += 1
                            else:
                                f.write("Failed: catid:" + str(current_target["labels"][row_id].numpy())
                                        + "," + str(current_target["boxes"][row_id].numpy()) + ", width: "
                                        + str(box_width.numpy())
                                        + ", Height: " + str(box_height.numpy()) + "\n")
                                if current_target["labels"][row_id] == 1:
                                    total_doors += 1
                                if current_target["labels"][row_id] == 3:
                                    total_stairs += 1
                                if current_target["labels"][row_id] == 4:
                                    total_ramps += 1
                else:
                    f.write("Undetected Boxes:\n")
                    # print("Undetected Boxes:")
                    for boxidx, gtbox in enumerate(current_target["boxes"]):
                        box_width = gtbox[2] - gtbox[0]
                        box_height = gtbox[3] - gtbox[1]
                        f.write("Failed: catid:" + str(current_target["labels"][boxidx].numpy()) + ","
                                + str(gtbox.numpy()) + ", width: " + str(box_width.numpy())
                                + ", Height: " + str(box_height.numpy()) + "\n")


        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # print("Result:" + str(res))

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    with open('test_result.txt', 'a') as f:
        f.write("TotalDet_DOORS: " + str(total_detected_doors) + "\n")
        f.write("TotalDet_KNOBS: " + str(total_detected_knobs) + "\n")
        f.write("TotalDet_STAIRS: " + str(total_detected_stairs) + "\n")
        f.write("TotalDet_RAMPS: " + str(total_detected_ramps) + "\n")
        f.write("Total_DOORS: " + str(total_doors) + "\n")
        f.write("Total_KNOBS: " + str(total_knobs) + "\n")
        f.write("Total_STAIRS: " + str(total_stairs) + "\n")
        f.write("Total_RAMPS: " + str(total_ramps) + "\n")
        f.write("TP_DOORS: " + str(tp_doors) + "\n")
        f.write("FP_DOORS: " + str(fp_doors) + "\n")
        f.write("TP_KNOBS: " + str(tp_knobs) + "\n")
        f.write("FP_KNOBS: " + str(fp_knobs) + "\n")
        f.write("TP_STAIRS: " + str(tp_stairs) + "\n")
        f.write("FP_STAIRS: " + str(fp_stairs) + "\n")
        f.write("TP_RAMPS: " + str(tp_ramps) + "\n")
        f.write("FP_RAMPS: " + str(fp_ramps) + "\n")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return coco_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def single_box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def drawboundingboxTP(image, gtboxes, predictedboxes, cat_id, category, image_id, gtcolor, predcolor, predlabels, iouthres, gtlabels, result_folder):
    font_size = int(image.size[0] * 20.0 / 2000)
    font = ImageFont.truetype("FreeSerif.ttf", size=font_size)
    draw = ImageDraw.Draw(image)
    cat_name = category[cat_id]["name"]
    tp_box_ids = []
    for i in range(len(predictedboxes)):
        pred_iou_rank = []
        if predlabels[i] == cat_id:
            for box_id, box in enumerate(gtboxes):
                box_iou_score = single_box_iou(box, predictedboxes[i])
                if box_iou_score > iouthres and gtlabels[box_id] == predlabels[i]:
                    pred_iou_rank.append(box_iou_score)
                    tp_box_ids.append(i)
            if len(pred_iou_rank) > 0:
                draw.rectangle(predictedboxes[i], outline=predcolor[cat_id],
                               width=int(image.size[0] / 500.0))
                text = cat_name + "_" + str(np.round(max(pred_iou_rank), 2))
                draw.text((predictedboxes[i][0] + 1, predictedboxes[i][1] - font_size - 1),
                          text=text, fill=(255, 255, 255), font=font)

    for i in range(len(gtboxes)):
        if gtlabels[i] == cat_id:
            draw.rectangle(gtboxes[i], outline=gtcolor[cat_id],
                           width=int(image.size[0] / 500.0))
            # text = "GT_" + cat_name
            # draw.text((gtboxes[i][0] + 1, gtboxes[i][1] - font_size - 1),
            #           text=text, fill=(255, 255, 255), font=font)
    image.save(os.path.join(result_folder, str(image_id) + "_" + str(cat_id) + "_TP_result.jpg"))
    return tp_box_ids


def drawboundingboxFP(image, gtboxes, predictedboxes, cat_id, category, image_id, gtcolor, predcolor, predlabels, iouthres, gtlabels, result_folder, tp_boxes):
    font_size = int(image.size[0] * 20.0 / 2000)
    font = ImageFont.truetype("FreeSerif.ttf", size=font_size)
    draw = ImageDraw.Draw(image)
    cat_name = category[cat_id]["name"]
    fp_box_ids = []
    for i in range(len(predictedboxes)):
        pred_iou_rank = []
        if predlabels[i] == cat_id:
            for box_id, box in enumerate(gtboxes):
                box_iou_score = single_box_iou(box, predictedboxes[i])
                if box_iou_score <= iouthres and gtlabels[box_id] == predlabels[i] and i not in tp_boxes:
                    draw.rectangle(predictedboxes[i], outline=predcolor[cat_id],
                                   width=int(image.size[0] / 500.0))
                    pred_iou_rank.append(box_iou_score)
                    fp_box_ids.append(i)
            if len(pred_iou_rank) > 0:
                text = cat_name + "_" + str(np.round(max(pred_iou_rank), 2))
                draw.text((predictedboxes[i][0] + 1, predictedboxes[i][1] - font_size - 1),
                          text=text, fill=(255, 255, 255), font=font)
    for i in range(len(gtboxes)):
        if gtlabels[i] == cat_id:
            draw.rectangle(gtboxes[i], outline=gtcolor[cat_id],
                           width=int(image.size[0] / 500.0))
            # text = "GT_" + cat_name
            # draw.text((gtboxes[i][0] + 1, gtboxes[i][1] - font_size - 1),
            #           text=text, fill=(255, 255, 255), font=font)
    image.save(os.path.join(result_folder, str(image_id) + "_" + str(cat_id) + "_FP_result.jpg"))
    return fp_box_ids


def delete_element(list_object, indices):
    list_object = list_object.tolist()
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

    return torch.tensor(list_object)


def processed_predictions(gtboxes, gtlabels, predboxes, predscores, predlabels):
    processed_boxes = []
    processed_scores = []
    processed_labels = []
    gt_doors_idx = [i for i, x in enumerate(gtlabels) if x == 1]
    knobs_idx = [i for i, x in enumerate(predlabels) if x == 2]
    removable_idx = []
    knob_iou_list = []
    if len(gt_doors_idx) > 0:
        if len(knobs_idx) > 0:
            if len(gt_doors_idx) == 1:
                door_box = gtboxes[gt_doors_idx[0]]
                for i in knobs_idx:
                    knob_iou = single_box_iou(door_box, predboxes[i])
                    if knob_iou == 0:
                        removable_idx.append(i)
            elif len(gt_doors_idx) > 1:
                if len(knobs_idx) == 1:
                    knob_box = predboxes[knobs_idx[0]]
                    for i in gt_doors_idx:
                        knob_iou_list.append(single_box_iou(gtboxes[i], knob_box))
                    if max(knob_iou_list) > 0:
                        removable_idx.append(knobs_idx[0])
                elif len(knobs_idx) > 1:
                    gt_door_box_list = torch.stack([gtboxes[i] for i in gt_doors_idx])
                    knob_box_list = torch.stack([predboxes[i] for i in knobs_idx])
                    iou_matrix = box_iou(knob_box_list, gt_door_box_list)
                    for row_id, row in enumerate(iou_matrix):
                        if row.max() == 0:
                            removable_idx.append(knobs_idx[row_id])

            predboxes = delete_element(predboxes, removable_idx)
            predscores = delete_element(predscores, removable_idx)
            predlabels = delete_element(predlabels, removable_idx)
    else:
        if len(knobs_idx) > 0:
            predboxes = delete_element(predboxes, knobs_idx)
            predscores = delete_element(predscores, knobs_idx)
            predlabels = delete_element(predlabels, knobs_idx)

    processed_boxes.append(predboxes)
    processed_labels.append(predscores)
    processed_scores.append(predlabels)
    assert len(predboxes) == len(predscores)
    assert len(predboxes) == len(predscores)

    return predboxes, predscores, predlabels

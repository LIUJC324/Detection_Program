from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch


@dataclass
class DetectionMetricResult:
    map50: float
    recall50: float
    small_recall50: float
    per_class_ap50: Dict[int, float]
    true_positives: int
    false_positives: int
    false_negatives: int
    per_class_stats: Dict[int, Dict[str, float]]
    threshold_metrics: Dict[float, Dict[str, float]]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2 - inter + 1e-6
    return inter / union


def _match_single_image(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
):
    if pred_boxes.numel() == 0:
        return [], torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)
    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    pred_labels = pred_labels[order]
    matched = []
    used_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        same_class = gt_labels == label
        if same_class.sum() == 0:
            matched.append((int(label.item()), float(score.item()), 0))
            continue
        class_indices = torch.where(same_class)[0]
        ious = box_iou(box.unsqueeze(0), gt_boxes[class_indices]).squeeze(0)
        best_iou, best_idx = torch.max(ious, dim=0)
        target_index = class_indices[best_idx]
        is_tp = int(best_iou.item() >= iou_threshold and not used_gt[target_index])
        if is_tp:
            used_gt[target_index] = True
        matched.append((int(label.item()), float(score.item()), is_tp))
    return matched, used_gt


def _ap_from_ranked(matches: Iterable[int], num_gt: int) -> float:
    if num_gt == 0:
        return 0.0
    matches_tensor = torch.as_tensor(list(matches), dtype=torch.float32)
    if matches_tensor.numel() == 0:
        return 0.0
    if matches_tensor.sum().item() <= 0:
        return 0.0
    tp = torch.cumsum(matches_tensor, dim=0)
    fp = torch.cumsum(1.0 - matches_tensor, dim=0)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / max(num_gt, 1)
    precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    for idx in range(precision.numel() - 1, 0, -1):
        precision[idx - 1] = torch.maximum(precision[idx - 1], precision[idx])
    changing_points = torch.where(recall[1:] != recall[:-1])[0]
    if changing_points.numel() == 0:
        return 0.0
    ap = torch.sum((recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1])
    return float(ap.item())


def _filter_predictions(
    predictions: List[Dict[str, torch.Tensor]],
    score_threshold: float,
) -> List[Dict[str, torch.Tensor]]:
    filtered_predictions: List[Dict[str, torch.Tensor]] = []
    for prediction in predictions:
        scores = prediction["scores"]
        keep = scores >= score_threshold
        filtered_predictions.append(
            {
                "boxes": prediction["boxes"][keep],
                "scores": prediction["scores"][keep],
                "labels": prediction["labels"][keep],
            }
        )
    return filtered_predictions


def _evaluate_core(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float,
    small_object_area: float,
) -> DetectionMetricResult:
    per_class_records: Dict[int, List[tuple]] = {cls_id: [] for cls_id in range(num_classes)}
    per_class_gt = {cls_id: 0 for cls_id in range(num_classes)}
    per_class_tp = {cls_id: 0 for cls_id in range(num_classes)}
    per_class_predictions = {cls_id: 0 for cls_id in range(num_classes)}
    per_class_score_sum = {cls_id: 0.0 for cls_id in range(num_classes)}
    total_gt = 0
    total_predictions = 0
    matched_gt = 0
    small_total = 0
    small_matched = 0

    for prediction, target in zip(predictions, targets):
        gt_boxes = target["boxes"].detach().cpu()
        gt_labels = target["labels"].detach().cpu()
        pred_boxes = prediction["boxes"].detach().cpu()
        pred_scores = prediction["scores"].detach().cpu()
        pred_labels = prediction["labels"].detach().cpu()

        matches, used_gt = _match_single_image(
            pred_boxes,
            pred_scores,
            pred_labels,
            gt_boxes,
            gt_labels,
            iou_threshold,
        )
        matched_count = int(used_gt.sum().item())
        matched_gt += matched_count
        total_gt += gt_boxes.shape[0]
        total_predictions += pred_boxes.shape[0]

        for cls_id in range(num_classes):
            gt_mask = gt_labels == cls_id
            pred_mask = pred_labels == cls_id
            per_class_gt[cls_id] += int(gt_mask.sum().item())
            per_class_predictions[cls_id] += int(pred_mask.sum().item())
            if pred_mask.any():
                per_class_score_sum[cls_id] += float(pred_scores[pred_mask].sum().item())
            if gt_mask.any() and used_gt.numel() > 0:
                per_class_tp[cls_id] += int((used_gt & gt_mask).sum().item())

        if gt_boxes.numel() > 0:
            areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            small_total += int((areas <= small_object_area).sum().item())

            used_small = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
            if pred_boxes.numel() > 0:
                order = torch.argsort(pred_scores, descending=True)
                for idx in order.tolist():
                    label = pred_labels[idx]
                    class_mask = gt_labels == label
                    if class_mask.sum() == 0:
                        continue
                    class_indices = torch.where(class_mask)[0]
                    ious = box_iou(pred_boxes[idx].unsqueeze(0), gt_boxes[class_indices]).squeeze(0)
                    best_iou, best_idx = torch.max(ious, dim=0)
                    target_index = class_indices[best_idx]
                    area = areas[target_index]
                    if (
                        best_iou.item() >= iou_threshold
                        and area.item() <= small_object_area
                        and not used_small[target_index]
                    ):
                        used_small[target_index] = True
                        small_matched += 1

        for label, score, is_tp in matches:
            per_class_records[label].append((score, is_tp))

    per_class_ap = {}
    per_class_stats: Dict[int, Dict[str, float]] = {}
    for cls_id, records in per_class_records.items():
        ranked = [item[1] for item in sorted(records, key=lambda x: x[0], reverse=True)]
        ap50 = _ap_from_ranked(ranked, per_class_gt[cls_id])
        tp = per_class_tp[cls_id]
        pred_count = per_class_predictions[cls_id]
        gt_count = per_class_gt[cls_id]
        fp = pred_count - tp
        fn = gt_count - tp
        avg_score = per_class_score_sum[cls_id] / max(pred_count, 1)
        per_class_ap[cls_id] = ap50
        per_class_stats[cls_id] = {
            "ap50": ap50,
            "num_gt": gt_count,
            "num_predictions": pred_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "avg_score": avg_score,
        }

    true_positives = matched_gt
    false_positives = total_predictions - true_positives
    false_negatives = total_gt - true_positives
    map50 = sum(per_class_ap.values()) / max(num_classes, 1)
    recall50 = true_positives / max(total_gt, 1)
    small_recall50 = small_matched / max(small_total, 1)
    return DetectionMetricResult(
        map50=map50,
        recall50=recall50,
        small_recall50=small_recall50,
        per_class_ap50=per_class_ap,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        per_class_stats=per_class_stats,
        threshold_metrics={},
    )


def evaluate_predictions(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
    small_object_area: float = 1024.0,
    score_thresholds: List[float] | None = None,
) -> DetectionMetricResult:
    result = _evaluate_core(
        predictions=predictions,
        targets=targets,
        num_classes=num_classes,
        iou_threshold=iou_threshold,
        small_object_area=small_object_area,
    )
    if score_thresholds:
        for threshold in sorted({float(item) for item in score_thresholds}):
            filtered_predictions = _filter_predictions(predictions, threshold)
            threshold_result = _evaluate_core(
                predictions=filtered_predictions,
                targets=targets,
                num_classes=num_classes,
                iou_threshold=iou_threshold,
                small_object_area=small_object_area,
            )
            result.threshold_metrics[threshold] = {
                "map50": threshold_result.map50,
                "recall50": threshold_result.recall50,
                "small_recall50": threshold_result.small_recall50,
                "true_positives": threshold_result.true_positives,
                "false_positives": threshold_result.false_positives,
                "false_negatives": threshold_result.false_negatives,
            }
    return result

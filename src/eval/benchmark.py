import os
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import numpy as np
import textdistance
import pandas as pd

# Define the entity types that can be extracted from documents
ENTITY_TYPES = ["Email Address", "Phone Number", "Contract Number", "Date", "Location", "Organization Name", "Person Name"]


def bb_intersection_over_union(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        boxA: First bounding box as [x1, y1, x2, y2]
        boxB: Second bounding box as [x1, y1, x2, y2]
    
    Returns:
        float: IoU value between 0 and 1, where 1 means perfect overlap
    """
    # determine the (x, y)-coordinates of the intersection rectangle
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
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def match_entities(predictions, ground_truth, iou_threshold=0.5, document_id=None):
    """
    Match predicted entities with ground truth entities using both text similarity and spatial overlap.
    
    This function implements a two-pass matching strategy:
    1. Exact match pass: Find entities with identical text values and sufficient spatial overlap
    2. Fuzzy match pass: Find entities with similar text values (using Jaro distance) and sufficient spatial overlap
    
    Args:
        predictions: List of predicted entities, each with 'type', 'value', and 'bbox' fields
        ground_truth: List of ground truth entities, each with 'type', 'value', and 'bbox' fields
        iou_threshold: Minimum IoU required for spatial overlap (default: 0.5)
        document_id: Optional document identifier to include in results (default: None)
    
    Returns:
        tuple: (tp_value, fp_value, fn_value, gt_counts, pred_counts, results_df, all_entity_types, bbox_fails, unmatched_gt_info) 
               - tp_value: True positives (correctly predicted entities)
               - fp_value: False positives (incorrectly predicted entities)
               - fn_value: False negatives (missed ground truth entities)
               - gt_counts: Number of ground truth entities per type
               - pred_counts: Number of predicted entities per type
               - results_df: DataFrame with detailed matching results including document_id
               - all_entity_types: List of all entity types processed
               - bbox_fails: Number of text matches that failed due to bbox threshold
               - unmatched_gt_info: Information about unmatched ground truth entries
    """
    
    # Initialize counters for True Positives, False Positives, and False Negatives
    tp_value = defaultdict(int)  # Correctly predicted entities
    fp_value = defaultdict(int)  # Incorrectly predicted entities (not in ground truth)
    fn_value = defaultdict(int)  # Missed ground truth entities (not predicted)
    gt_counts = defaultdict(int)   # Total number of ground truth entities per type
    pred_counts = defaultdict(int) # Total number of predicted entities per type
    bbox_fails = defaultdict(int)  # Text matches that failed due to bbox threshold
    unmatched_gt_info = defaultdict(list)  # Store info about unmatched ground truth entries

    # Group entities by type for easier processing
    ground_truth_dict = defaultdict(list)
    predictions_dict = defaultdict(list)

    # Here we define a pandas dataframe to store the results with explicit dtypes
    results_df = pd.DataFrame(columns=pd.Index([
        "Document ID",
        "Entity Type",
        "Ground Truth",
        "Predicted",
        "Probability",
        "is_correct",
        "annotators",
        "bbox_iou",
        "text_similarity",
        "best_match_info",
        "gt_bbox",
        "pred_bbox",
        "votes"
    ]))
    # Set explicit dtypes to avoid FutureWarning
    results_df = results_df.astype({
        "Document ID": "string",
        "Entity Type": "string",
        "Ground Truth": "string", 
        "Predicted": "string",
        "Probability": "float64",
        "is_correct": "boolean",
        "annotators": "string",
        "bbox_iou": "float64",
        "text_similarity": "float64",
        "best_match_info": "string",
        "votes": "object"
    })

    # Organize ground truth entities by their type
    for entity in ground_truth:
        if isinstance(entity["type"], list):
            ground_truth_dict[entity["type"][0]].append(entity)
        else:
            ground_truth_dict[entity["type"]].append(entity)

    # Organize predicted entities by their type
    for entity in predictions:
        if isinstance(entity["type"], list):
            predictions_dict[entity["type"][0]].append(entity)
        else:
            predictions_dict[entity["type"]].append(entity)

    # Process each entity type separately
    all_entity_types = ENTITY_TYPES
    for entity_type in all_entity_types:
        ground_truth_entities = ground_truth_dict.get(entity_type, [])
        predictions_entities = predictions_dict.get(entity_type, [])

        # Store the total counts for this entity type
        gt_counts[entity_type] = len(ground_truth_entities)
        pred_counts[entity_type] = len(predictions_entities)

        # Extract text values and bounding boxes for matching
        ground_truth_values = [e["value"] for e in ground_truth_entities]
        predictions_values = [e["value"] for e in predictions_entities]

        # Track which entities have been matched to avoid duplicate matches
        matched_gt = set()   # Indices of matched ground truth entities
        matched_pred = set() # Indices of matched predicted entities

        # Keep track of best matches for each ground truth entry
        best_matches = {}  # gt_idx -> (pred_idx, sim, iou)

        # Threshold for fuzzy text matching using Jaro distance
        threshold = 0.85

        # Two-pass matching strategy: exact match first, then fuzzy match
        for pass_type, sim_thresh in [("exact", 1.0), ("fuzzy", threshold)]:
            for i, pred_val in enumerate(predictions_values):
                if i in matched_pred:
                    continue
                
                pred_val_normalized = str(pred_val).lower().strip()
                pred_bbox = predictions_entities[i]["bbox"]

                best_match_j = None
                best_similarity = 0.0
                best_iou = 0.0

                for j, gt_val in enumerate(ground_truth_values):
                    if j in matched_gt:
                        continue

                    gt_val_normalized = str(gt_val).lower().strip()
                    gt_bbox = ground_truth_entities[j]["bbox"]
                    
                    sim = textdistance.jaro(pred_val_normalized, gt_val_normalized)
                    iou = bb_intersection_over_union(pred_bbox, gt_bbox)

                    # Update best match info for this ground truth entry
                    if j not in best_matches or (sim * 0.1 + iou * 0.9) > (best_matches[j][1] * 0.1 + best_matches[j][2] * 0.9):
                        best_matches[j] = (i, sim, iou)
                    
                    if sim >= sim_thresh and iou >= iou_threshold:
                        combined_score = sim * 0.1 + iou * 0.9
                        if best_match_j is None or combined_score > (best_similarity * 0.1 + best_iou * 0.9):
                            best_match_j = j
                            best_similarity = sim
                            best_iou = iou

                if best_match_j is not None and best_iou >= iou_threshold:
                    matched_gt.add(best_match_j)
                    matched_pred.add(i)
                    tp_value[entity_type] += 1
                    
                    gt_val_original = ground_truth_values[best_match_j]
                    gt_val_normalized = str(gt_val_original).lower()
                    
                    new_row = pd.DataFrame([{
                        "Document ID": document_id,
                        "Entity Type": entity_type,
                        "Ground Truth": gt_val_normalized,
                        "Predicted": pred_val_normalized,
                        "Probability": float(predictions_entities[i]["probability"]),
                        "is_correct": True,
                        "annotators": str(predictions_entities[i]["annotators"]),
                        "votes": predictions_entities[i]["votes"],
                        "bbox_iou": best_iou,
                        "text_similarity": best_similarity,
                        "best_match_info": "match",
                        "gt_bbox": ground_truth_entities[best_match_j]["bbox"],
                        "pred_bbox": predictions_entities[i]["bbox"]
                    }], dtype=object)
                    results_df = pd.concat([results_df, new_row], ignore_index=True)

                elif best_match_j is not None and best_iou < iou_threshold:
                    # Text matched but bbox failed — count once per prediction
                    bbox_fails[entity_type] += 1

                    gt_val_original = ground_truth_values[best_match_j]
                    gt_val_normalized = str(gt_val_original).lower()
                    new_row = pd.DataFrame([{
                        "Document ID": document_id,
                        "Entity Type": entity_type,
                        "Ground Truth": gt_val_normalized,
                        "Predicted": pred_val_normalized,
                        "Probability": float(predictions_entities[i]["probability"]),
                        "is_correct": False,
                        "annotators": str(predictions_entities[i]["annotators"]),
                        "votes": predictions_entities[i]["votes"],
                        "bbox_iou": best_iou,
                        "text_similarity": best_similarity,
                        "best_match_info": "bbox_fail",
                        "gt_bbox": ground_truth_entities[best_match_j]["bbox"],
                        "pred_bbox": predictions_entities[i]["bbox"]
                    }], dtype=object)
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
                    # Do NOT mark as matched; prediction will be treated as false positive later

        # Add False Positives to results dataframe
        for i, pred_val in enumerate(predictions_values):
            if i not in matched_pred:
                new_row = pd.DataFrame([{
                    "Document ID": document_id,
                    "Entity Type": entity_type,
                    "Ground Truth": "",
                    "Predicted": str(predictions_values[i]),
                    "Probability": float(predictions_entities[i]["probability"]),
                    "is_correct": False,
                    "annotators": str(predictions_entities[i]["annotators"]),
                    "votes": predictions_entities[i]["votes"],
                    "bbox_iou": 0.0,
                    "text_similarity": 0.0,
                    "best_match_info": "false_positive",
                    "gt_bbox": None,
                    "pred_bbox": predictions_entities[i]["bbox"]
                }], dtype=object)
                results_df = pd.concat([results_df, new_row], ignore_index=True)
    
        # Add False Negatives to results dataframe and collect unmatched GT info
        for j, gt_val in enumerate(ground_truth_values):
            if j not in matched_gt:
                best_match_info = ""
                if j in best_matches:
                    pred_idx, best_sim, best_iou = best_matches[j]
                    pred_val = predictions_values[pred_idx]
                    best_match_info = f"Best match: '{pred_val}' (sim={best_sim:.3f}, iou={best_iou:.3f})"
                    # Store detailed info about the unmatched GT entry
                    unmatched_gt_info[entity_type].append({
                        'gt_value': gt_val,
                        'best_match_value': pred_val,
                        'text_similarity': best_sim,
                        'bbox_iou': best_iou,
                        'document_id': document_id
                    })
                
                new_row = pd.DataFrame([{
                    "Document ID": document_id,
                    "Entity Type": entity_type,
                    "Ground Truth": str(ground_truth_values[j]),
                    "Predicted": "",
                    "Probability": 0.0,
                    "is_correct": False,
                    "annotators": "ground_truth",
                    "votes": [],
                    "bbox_iou": best_matches.get(j, (None, 0, 0))[2],
                    "text_similarity": best_matches.get(j, (None, 0, 0))[1],
                    "best_match_info": "false_negative",
                    "gt_bbox": ground_truth_entities[j]["bbox"],
                    "pred_bbox": None
                }], dtype=object)
                results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Calculate false positives and false negatives for this entity type
        fp_value[entity_type] = len(predictions_values) - len(matched_pred)  # Unmatched predictions
        fn_value[entity_type] = len(ground_truth_values) - len(matched_gt)   # Unmatched ground truth

    return tp_value, fp_value, fn_value, gt_counts, pred_counts, results_df, all_entity_types, bbox_fails, unmatched_gt_info


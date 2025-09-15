import os
import json
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import textdistance
import pandas as pd
from typing import Dict, List, Tuple, Any
import argparse


# Define the entity types that can be extracted from documents
# ENTITY_TYPES = ["Email Address", "Invoice Number", "Phone Number", "Contract Number", "Date", "Location", "Organization Name", "Person Name"]
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


def match_entities_simplified(predictions, ground_truth, iou_threshold=0.5, document_id=None):
    """
    Match predicted entities with ground truth entities using both text similarity and spatial overlap.
    This is a simplified version that excludes: annotators, best_match_info, gt_bbox, pred_bbox, votes
    
    Args:
        predictions: List of predicted entities, each with 'type', 'value', and optionally 'bbox' fields
        ground_truth: List of ground truth entities, each with 'type', 'value', and optionally 'bbox' fields
        iou_threshold: Minimum IoU required for spatial overlap (default: 0.5). If 0.0, bbox matching is skipped.
        document_id: Optional document identifier to include in results (default: None)
    
    Returns:
        tuple: (tp_value, fp_value, fn_value, gt_counts, pred_counts, results_df, all_entity_types, bbox_fails, unmatched_gt_info)
    """
    
    # Initialize counters
    tp_value = defaultdict(int)
    fp_value = defaultdict(int)
    fn_value = defaultdict(int)
    gt_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    bbox_fails = defaultdict(int)
    unmatched_gt_info = defaultdict(list)

    # Group entities by type
    ground_truth_dict = defaultdict(list)
    predictions_dict = defaultdict(list)

    # Create simplified results dataframe (excluding specified columns)
    results_df = pd.DataFrame(columns=pd.Index([
        "Document ID",
        "Entity Type",
        "Ground Truth",
        "Predicted",
        "is_correct",
        "bbox_iou",
        "text_similarity"
    ]))
    
    # Set explicit dtypes
    results_df = results_df.astype({
        "Document ID": "string",
        "Entity Type": "string",
        "Ground Truth": "string", 
        "Predicted": "string",
        "is_correct": "boolean",
        "bbox_iou": "float64",
        "text_similarity": "float64"
    })

    # Organize ground truth entities by their type
    for entity in ground_truth:
        if isinstance(entity["type"], list):
            ground_truth_dict[entity["type"][0]].append(entity)
        else:
            ground_truth_dict[entity["type"]].append(entity)

    # Organize predicted entities by their type
    for entity in predictions:
        if isinstance(entity["entity_type"], list):
            predictions_dict[entity["entity_type"][0]].append(entity)
        else:
            predictions_dict[entity["entity_type"]].append(entity)

    # Process each entity type separately
    # Get all entity types that exist in either ground truth or predictions
    gt_entity_types = set()
    for entity in ground_truth:
        if "type" in entity:
            entity_type = entity["type"]
            if isinstance(entity_type, list):
                gt_entity_types.update(entity_type)
            else:
                gt_entity_types.add(entity_type)
    
    pred_entity_types = set()
    for entity in predictions:
        if "entity_type" in entity:
            entity_type = entity["entity_type"]
            if isinstance(entity_type, list):
                pred_entity_types.update(entity_type)
            else:
                pred_entity_types.add(entity_type)
    
    all_entity_types = list(gt_entity_types | pred_entity_types | set(ENTITY_TYPES))
    
    for entity_type in all_entity_types:
        ground_truth_entities = ground_truth_dict.get(entity_type, [])
        predictions_entities = predictions_dict.get(entity_type, [])

        # Store the total counts for this entity type
        gt_counts[entity_type] = len(ground_truth_entities)
        pred_counts[entity_type] = len(predictions_entities)

        # Extract text values and bounding boxes for matching
        ground_truth_values = [e["value"] for e in ground_truth_entities]
        predictions_values = [e["value"] for e in predictions_entities]

        # Track which entities have been matched
        matched_gt = set()
        matched_pred = set()

        # Keep track of best matches for each ground truth entry
        best_matches = {}

        # Threshold for fuzzy text matching using Jaro distance
        threshold = 0.85

        # Two-pass matching strategy: exact match first, then fuzzy match
        for pass_type, sim_thresh in [("exact", 1.0), ("fuzzy", threshold)]:
            for i, pred_val in enumerate(predictions_values):
                if i in matched_pred:
                    continue
                
                pred_val_normalized = str(pred_val).lower().strip()
                # Handle missing bbox fields - use dummy bbox if not present or iou_threshold is 0
                pred_bbox = predictions_entities[i].get("bbox", [0, 0, 1, 1])

                best_match_j = None
                best_similarity = 0.0
                best_iou = 0.0

                for j, gt_val in enumerate(ground_truth_values):
                    if j in matched_gt:
                        continue

                    gt_val_normalized = str(gt_val).lower().strip()
                    # Handle missing bbox fields - use dummy bbox if not present or iou_threshold is 0
                    gt_bbox = ground_truth_entities[j].get("bbox", [0, 0, 1, 1])
                    
                    sim = textdistance.jaro(pred_val_normalized, gt_val_normalized)
                    
                    # Skip IoU calculation if threshold is 0.0 or bbox is missing
                    if iou_threshold == 0.0 or "bbox" not in predictions_entities[i] or "bbox" not in ground_truth_entities[j]:
                        iou = 1.0  # Set IoU to 1.0 to bypass IoU constraint
                    else:
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
                    
                    # Simplified row without excluded columns
                    new_row = pd.DataFrame([{
                        "Document ID": document_id,
                        "Entity Type": entity_type,
                        "Ground Truth": gt_val_normalized,
                        "Predicted": pred_val_normalized,
                        "is_correct": True,
                        "bbox_iou": best_iou,
                        "text_similarity": best_similarity
                    }], dtype=object)
                    results_df = pd.concat([results_df, new_row], ignore_index=True)

                elif best_match_j is not None and best_iou < iou_threshold:
                    # Text matched but bbox failed
                    bbox_fails[entity_type] += 1

                    gt_val_original = ground_truth_values[best_match_j]
                    gt_val_normalized = str(gt_val_original).lower()
                    new_row = pd.DataFrame([{
                        "Document ID": document_id,
                        "Entity Type": entity_type,
                        "Ground Truth": gt_val_normalized,
                        "Predicted": pred_val_normalized,
                        "is_correct": False,
                        "bbox_iou": best_iou,
                        "text_similarity": best_similarity
                    }], dtype=object)
                    results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Add False Positives to results dataframe
        for i, pred_val in enumerate(predictions_values):
            if i not in matched_pred:
                new_row = pd.DataFrame([{
                    "Document ID": document_id,
                    "Entity Type": entity_type,
                    "Ground Truth": "",
                    "Predicted": str(predictions_values[i]),
                    "is_correct": False,
                    "bbox_iou": 0.0,
                    "text_similarity": 0.0
                }], dtype=object)
                results_df = pd.concat([results_df, new_row], ignore_index=True)
    
        # Add False Negatives to results dataframe and collect unmatched GT info
        for j, gt_val in enumerate(ground_truth_values):
            if j not in matched_gt:
                if j in best_matches:
                    pred_idx, best_sim, best_iou = best_matches[j]
                    pred_val = predictions_values[pred_idx]
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
                    "is_correct": False,
                    "bbox_iou": best_matches.get(j, (None, 0, 0))[2],
                    "text_similarity": best_matches.get(j, (None, 0, 0))[1]
                }], dtype=object)
                results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Calculate false positives and false negatives for this entity type
        fp_value[entity_type] = len(predictions_values) - len(matched_pred)
        fn_value[entity_type] = len(ground_truth_values) - len(matched_gt)

    return tp_value, fp_value, fn_value, gt_counts, pred_counts, results_df, all_entity_types, bbox_fails, unmatched_gt_info


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file and return its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def benchmark_directories(test_set_labels_dir: str, pred_labels_dir: str, iou_threshold: float = 0.5):
    """
    Benchmark predictions against ground truth labels from two directories.
    
    Args:
        test_set_labels_dir: Directory containing ground truth JSON files
        pred_labels_dir: Directory containing prediction JSON files
        iou_threshold: IoU threshold for bounding box matching
    
    Returns:
        pandas.DataFrame: Combined results from all processed files
    """
    
    # Get all JSON files from both directories
    test_files = {f for f in os.listdir(test_set_labels_dir) if f.endswith('.json')}
    pred_files = {f for f in os.listdir(pred_labels_dir) if f.endswith('.json')}
    
    # Find common files
    common_files = test_files.intersection(pred_files)
    
    if not common_files:
        print("No common JSON files found between directories!")
        return pd.DataFrame()
    
    print(f"Found {len(common_files)} common files to process")
    
    # Initialize combined results
    all_results = []
    overall_tp = defaultdict(int)
    overall_fp = defaultdict(int)
    overall_fn = defaultdict(int)
    overall_gt_counts = defaultdict(int)
    overall_pred_counts = defaultdict(int)
    overall_bbox_fails = defaultdict(int)
    overall_unmatched_gt_info = defaultdict(list)
    
    # Process each common file
    for filename in sorted(common_files):
        print(f"Processing: {filename}")
        
        # Load ground truth and predictions
        test_file_path = os.path.join(test_set_labels_dir, filename)
        pred_file_path = os.path.join(pred_labels_dir, filename)
        
        ground_truth_data = load_json_file(test_file_path)
        predictions_data = load_json_file(pred_file_path)
        
        if not ground_truth_data or not predictions_data:
            print(f"Skipping {filename} due to loading error")
            continue
        
        # Extract entities from the data
        # Assuming the JSON structure has an 'entities' key
        ground_truth_entities = ground_truth_data
        predictions_entities = predictions_data
        
        # Use filename (without extension) as document ID
        document_id = os.path.splitext(filename)[0]
        
        # Perform matching
        tp_value, fp_value, fn_value, gt_counts, pred_counts, results_df, all_entity_types, bbox_fails, unmatched_gt_info = match_entities_simplified(
            predictions_entities, ground_truth_entities, iou_threshold, document_id
        )
        
        # Add to overall results
        all_results.append(results_df)
        
        # Accumulate overall statistics
        for entity_type in all_entity_types:
            overall_tp[entity_type] += tp_value[entity_type]
            overall_fp[entity_type] += fp_value[entity_type]
            overall_fn[entity_type] += fn_value[entity_type]
            overall_gt_counts[entity_type] += gt_counts[entity_type]
            overall_pred_counts[entity_type] += pred_counts[entity_type]
            overall_bbox_fails[entity_type] += bbox_fails[entity_type]
            overall_unmatched_gt_info[entity_type].extend(unmatched_gt_info[entity_type])
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Print summary statistics
        print("\n=== BENCHMARK RESULTS ===")
        print(f"Total files processed: {len(all_results)}")
        print(f"Total rows in results: {len(combined_results)}")
        
        # Calculate and print metrics by entity type
        print("\nMetrics by Entity Type:")
        print("-" * 80)
        print(f"{'Entity Type':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
        print("-" * 80)
        
        for entity_type in ENTITY_TYPES:
            tp = overall_tp[entity_type]
            fp = overall_fp[entity_type]
            fn = overall_fn[entity_type]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Round metrics to 2 decimal places
            precision = round(precision, 2)
            recall = round(recall, 2)
            f1 = round(f1, 2)
            
            print(f"{entity_type:<20} {precision:<10.2f} {recall:<10.2f} {f1:<10.2f} {tp:<5} {fp:<5} {fn:<5}")
        
        # Calculate overall MACRO metrics across all predefined entity types
        supported_types = list(ENTITY_TYPES)
        
        macro_precisions = []
        macro_recalls = []
        macro_f1s = []
        for et in supported_types:
            tp = overall_tp[et]
            fp = overall_fp[et]
            fn = overall_fn[et]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            macro_precisions.append(p)
            macro_recalls.append(r)
            macro_f1s.append(f1)
        overall_precision = round(sum(macro_precisions) / len(macro_precisions), 2) if macro_precisions else 0.0
        overall_recall = round(sum(macro_recalls) / len(macro_recalls), 2) if macro_recalls else 0.0
        overall_f1 = round(sum(macro_f1s) / len(macro_f1s), 2) if macro_f1s else 0.0
        
        # Also compute totals for reference
        total_tp = sum(overall_tp.values())
        total_fp = sum(overall_fp.values())
        total_fn = sum(overall_fn.values())
        
        print("-" * 80)
        print(f"{'OVERALL (macro)':<20} {overall_precision:<10.2f} {overall_recall:<10.2f} {overall_f1:<10.2f} {total_tp:<5} {total_fp:<5} {total_fn:<5}")
        print("-" * 80)
        
        return combined_results
    else:
        print("No results to combine!")
        return pd.DataFrame()


def benchmark_single_file(test_data_file: str, pred_labels_dir: str, iou_threshold: float = 0.5, return_metrics: bool = False, dataset_name: str = "ad_buy"):
    """
    Benchmark predictions against ground truth labels from a single JSON file.
    
    Args:
        test_data_file: Path to JSON file containing all ground truth data in format:
                       [{"file_name": "...", "labels": [...]}, ...]
        pred_labels_dir: Directory containing prediction JSON files
        iou_threshold: IoU threshold for bounding box matching
    
    Returns:
        pandas.DataFrame: Combined results from all processed files
    """

    if dataset_name == "ad_buy":
        ENTITY_TYPES = ["Email Address", "Invoice Number", "Phone Number", "Contract Number", "Date", "Location", "Organization Name", "Person Name"]
    else:
        ENTITY_TYPES = ["Email Address", "Phone Number", "Contract Number", "Date", "Location", "Organization Name", "Person Name"]
    
    # Load the test data file
    print(f"Loading test data from: {test_data_file}")
    test_data = load_json_file(test_data_file)
    
    if not test_data:
        print("Failed to load test data file!")
        return pd.DataFrame()
    
    if not isinstance(test_data, list):
        print("Test data should be a list of objects!")
        return pd.DataFrame()
    
    # Get all prediction files
    pred_files = {f for f in os.listdir(pred_labels_dir) if f.endswith('.json')}
    
    # Create a mapping from file_name to ground truth labels
    ground_truth_map = {}
    for item in test_data:
        if 'file_name' in item and 'labels' in item:
            file_name = item['file_name']
            # Convert labels to the expected format (list of entities with type, value, bbox)
            entities = []
            for label in item['labels']:
                if 'entity_type' in label and 'value' in label:
                    entity = {
                        'type': label['entity_type'],
                        'value': label['value']
                    }
                    # Add bbox if present
                    if 'bbox' in label:
                        entity['bbox'] = label['bbox']
                    entities.append(entity)
            ground_truth_map[file_name] = entities
    
    # Process ALL ground truth files, whether predictions exist or not
    all_gt_files = list(ground_truth_map.keys())
    files_with_predictions = []
    files_without_predictions = []
    
    for file_name in all_gt_files:
        pred_file = f"{file_name}.json"
        if pred_file in pred_files:
            files_with_predictions.append((file_name, pred_file))
        else:
            files_without_predictions.append(file_name)
    
    print(f"Found {len(all_gt_files)} ground truth files to process")
    print(f"Files with predictions: {len(files_with_predictions)}")
    print(f"Files without predictions: {len(files_without_predictions)} (will be treated as complete prediction failures)")
    
    # FIRST: Count ALL ground truth entities across all files
    print("\nCounting all ground truth entities...")
    all_gt_entities_by_type = defaultdict(int)
    for file_name in all_gt_files:
        ground_truth_entities = ground_truth_map[file_name]
        print(f"  {file_name}: {len(ground_truth_entities)} entities")
        
        # Count entities by type
        for entity in ground_truth_entities:
            if 'type' in entity:
                entity_type = entity['type']
                all_gt_entities_by_type[entity_type] += 1
    
    print(f"\nTotal Ground Truth Distribution:")
    print("-" * 50)
    print(f"{'Entity Type':<20} {'GT Count':<10}")
    print("-" * 50)
    total_gt_entities = 0
    for entity_type in sorted(all_gt_entities_by_type.keys()):
        gt_count = all_gt_entities_by_type[entity_type]
        print(f"{entity_type:<20} {gt_count:<10}")
        total_gt_entities += gt_count
    print("-" * 50)
    print(f"{'TOTAL':<20} {total_gt_entities:<10}")
    print("-" * 50)
    
    # Initialize combined results
    all_results = []
    overall_tp = defaultdict(int)
    overall_fp = defaultdict(int)
    overall_fn = defaultdict(int)
    overall_gt_counts = defaultdict(int)
    overall_pred_counts = defaultdict(int)
    overall_bbox_fails = defaultdict(int)
    overall_unmatched_gt_info = defaultdict(list)
    
    # Initialize GT counts with the actual counts we just calculated
    for entity_type, count in all_gt_entities_by_type.items():
        overall_gt_counts[entity_type] = count
    
    # Process files WITH predictions
    for file_name, pred_file in sorted(files_with_predictions):
        print(f"Processing: {file_name}")
        
        # Load predictions
        pred_file_path = os.path.join(pred_labels_dir, pred_file)
        predictions_data = load_json_file(pred_file_path)
        
        if not predictions_data:
            print(f"Skipping {file_name} due to prediction loading error")
            continue
        
        # Get ground truth entities
        ground_truth_entities = ground_truth_map[file_name]
        
        # Convert predictions to expected format if needed
        predictions_entities = predictions_data
        if isinstance(predictions_data, dict) and 'entities' in predictions_data:
            predictions_entities = predictions_data['entities']
        
        # Ensure predictions have the right format
        # Note: match_entities_simplified expects predictions to have 'entity_type' field
        formatted_predictions = []
        for pred in predictions_entities:
            if isinstance(pred, dict):
                # Handle different possible formats
                entity = {}
                if 'type' in pred:
                    entity['entity_type'] = pred['type']
                elif 'entity_type' in pred:
                    entity['entity_type'] = pred['entity_type']
                
                if 'value' in pred:
                    entity['value'] = pred['value']
                elif 'text' in pred:
                    entity['value'] = pred['text']
                
                if 'bbox' in pred:
                    entity['bbox'] = pred['bbox']
                elif 'bounding_box' in pred:
                    entity['bbox'] = pred['bounding_box']
                
                # Only require entity_type and value, bbox is optional
                if all(key in entity for key in ['entity_type', 'value']):
                    formatted_predictions.append(entity)
        
        # Use filename as document ID
        document_id = file_name
        
        # Perform matching
        tp_value, fp_value, fn_value, gt_counts, pred_counts, results_df, all_entity_types, bbox_fails, unmatched_gt_info = match_entities_simplified(
            formatted_predictions, ground_truth_entities, iou_threshold, document_id
        )
        
        # Add to overall results
        all_results.append(results_df)
        
        # Accumulate overall statistics
        # NOTE: overall_gt_counts is already set correctly upfront, don't add to it
        for entity_type in all_entity_types:
            overall_tp[entity_type] += tp_value[entity_type]
            overall_fp[entity_type] += fp_value[entity_type]
            overall_fn[entity_type] += fn_value[entity_type]
            # overall_gt_counts[entity_type] += gt_counts[entity_type]  # Don't add - already set correctly
            overall_pred_counts[entity_type] += pred_counts[entity_type]
            overall_bbox_fails[entity_type] += bbox_fails[entity_type]
            overall_unmatched_gt_info[entity_type].extend(unmatched_gt_info[entity_type])
    
    # Process files WITHOUT predictions (complete prediction failures)
    for file_name in sorted(files_without_predictions):
        print(f"Processing: {file_name} (NO PREDICTIONS - treating as complete failure)")
        
        # Get ground truth entities
        ground_truth_entities = ground_truth_map[file_name]
        print(f"  Ground truth entities in {file_name}: {len(ground_truth_entities)}")
        
        # Debug: print entity types in this file
        entity_types_in_file = set(entity['type'] for entity in ground_truth_entities if 'type' in entity)
        print(f"  Entity types in {file_name}: {entity_types_in_file}")
        
        # No predictions means empty predictions list
        formatted_predictions = []
        
        # Use filename as document ID
        document_id = file_name
        
        # Perform matching (will result in all ground truth being false negatives)
        tp_value, fp_value, fn_value, gt_counts, pred_counts, results_df, all_entity_types, bbox_fails, unmatched_gt_info = match_entities_simplified(
            formatted_predictions, ground_truth_entities, iou_threshold, document_id
        )
        
        # Debug: print gt_counts returned from match_entities_simplified
        print(f"  GT counts from match_entities_simplified: {dict(gt_counts)}")
        
        # Add to overall results
        all_results.append(results_df)
        
        # Accumulate overall statistics
        # NOTE: overall_gt_counts is already set correctly upfront, don't add to it
        for entity_type in all_entity_types:
            overall_tp[entity_type] += tp_value[entity_type]
            overall_fp[entity_type] += fp_value[entity_type]
            overall_fn[entity_type] += fn_value[entity_type]
            # overall_gt_counts[entity_type] += gt_counts[entity_type]  # Don't add - already set correctly
            overall_pred_counts[entity_type] += pred_counts[entity_type]
            overall_bbox_fails[entity_type] += bbox_fails[entity_type]
            overall_unmatched_gt_info[entity_type].extend(unmatched_gt_info[entity_type])
        
        # Debug: print current overall_gt_counts after this file
        print(f"  Overall GT counts after {file_name}: {dict(overall_gt_counts)}")
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Print summary statistics
        print(f"\n=== BENCHMARK RESULTS ===")
        print(f"Total files processed: {len(all_results)}")
        print(f"Total rows in results: {len(combined_results)}")
        
        # Calculate and display metrics by entity type
        all_entity_types = set(overall_tp.keys()) | set(overall_fp.keys()) | set(overall_fn.keys())
        
        metrics = {
            'by_type': [],
            'overall': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0},
            'counts': {
                'tp': dict(overall_tp),
                'fp': dict(overall_fp),
                'fn': dict(overall_fn),
                'gt': dict(overall_gt_counts),
                'pred': dict(overall_pred_counts),
                'bbox_fails': dict(overall_bbox_fails),
            }
        }

        if all_entity_types:
            print(f"\nMetrics by Entity Type:")
            print("-" * 90)
            print(f"{'Entity Type':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<5} {'FP':<5} {'FN':<5} {'GT':<5}")
            print("-" * 90)
            
            overall_tp_sum = 0
            overall_fp_sum = 0 
            overall_fn_sum = 0
            overall_gt_sum = 0
            
            for entity_type in sorted(all_entity_types):
                tp = overall_tp[entity_type]
                fp = overall_fp[entity_type]
                fn = overall_fn[entity_type]
                gt_count = overall_gt_counts[entity_type]
                
                precision_raw = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall_raw = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_raw = 2 * precision_raw * recall_raw / (precision_raw + recall_raw) if (precision_raw + recall_raw) > 0 else 0.0

                # Print rounded
                precision = round(precision_raw, 2)
                recall = round(recall_raw, 2)
                f1 = round(f1_raw, 2)
                
                print(f"{entity_type:<20} {precision:<10.2f} {recall:<10.2f} {f1:<10.2f} {tp:<5} {fp:<5} {fn:<5} {gt_count:<5}")

                metrics['by_type'].append({
                    'entity_type': entity_type,
                    'precision': round(precision_raw, 4),
                    'recall': round(recall_raw, 4),
                    'f1': round(f1_raw, 4),
                    'tp': int(tp),
                    'fp': int(fp),
                    'fn': int(fn),
                    'gt': int(gt_count),
                })
                
                overall_tp_sum += tp
                overall_fp_sum += fp
                overall_fn_sum += fn
                overall_gt_sum += gt_count
            
            # Calculate overall MACRO metrics across all predefined ENTITY_TYPES
            supported_types = list(ENTITY_TYPES)
            macro_precisions = []
            macro_recalls = []
            macro_f1s = []
            for t in supported_types:
                tp_t = overall_tp[t]
                fp_t = overall_fp[t]
                fn_t = overall_fn[t]
                p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
                r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                macro_precisions.append(p)
                macro_recalls.append(r)
                macro_f1s.append(f1)
            overall_precision = round(sum(macro_precisions) / len(macro_precisions), 2) if macro_precisions else 0.0
            overall_recall = round(sum(macro_recalls) / len(macro_recalls), 2) if macro_recalls else 0.0
            overall_f1 = round(sum(macro_f1s) / len(macro_f1s), 2) if macro_f1s else 0.0
            
            print("-" * 90)
            print(f"{'OVERALL (macro)':<20} {overall_precision:<10.2f} {overall_recall:<10.2f} {overall_f1:<10.2f} {overall_tp_sum:<5} {overall_fp_sum:<5} {overall_fn_sum:<5} {overall_gt_sum:<5}")
            print("-" * 90)

            metrics['overall'] = {
                'precision': round(sum(macro_precisions) / len(macro_precisions), 4) if macro_precisions else 0.0,
                'recall': round(sum(macro_recalls) / len(macro_recalls), 4) if macro_recalls else 0.0,
                'f1': round(sum(macro_f1s) / len(macro_f1s), 4) if macro_f1s else 0.0,
                'tp': int(overall_tp_sum),
                'fp': int(overall_fp_sum),
                'fn': int(overall_fn_sum),
                'gt': int(overall_gt_sum),
            }
        
        if return_metrics:
            return combined_results, metrics
        else:
            return combined_results
    
    else:
        print("No results to combine!")
        return pd.DataFrame()


def main():
    """Main function to run the directory benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark entity extraction predictions against ground truth labels')
    parser.add_argument('test_set_labels_dir', help='Directory containing ground truth JSON files')
    parser.add_argument('pred_labels_dir', help='Directory containing prediction JSON files')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for bounding box matching (default: 0.5)')
    parser.add_argument('--output', '-o', help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.test_set_labels_dir):
        print(f"Error: Test set directory does not exist: {args.test_set_labels_dir}")
        return
    
    if not os.path.exists(args.pred_labels_dir):
        print(f"Error: Predictions directory does not exist: {args.pred_labels_dir}")
        return
    
    # Run benchmark
    results_df = benchmark_directories(args.test_set_labels_dir, args.pred_labels_dir, args.iou_threshold)
    
    # Save results if output path is provided
    if args.output and not results_df.empty:
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

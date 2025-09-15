import os
import json
import numpy as np
from typing import List, Dict, Any
from typing import List, Dict, Any
import random
from collections import defaultdict
import textdistance


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
                    file_name = file_name.split(".")[0] + ".json"
                    file_dimensions_map[file_name] = {
                        "original_width": width,
                        "original_height": height
                    }

    return file_dimensions_map

dimensions_json_path = "//Volumes/MyDataDrive/thesis/code-2/data/manual-label-2.json"
dimensions_map = load_dimensions_map(dimensions_json_path)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def bb_intersection_over_union(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def majority_vote(item: Dict[str, Any], max_annotators: int) -> float:
    """
    Majority vote for a item.
    """
    if not item:
        return 0.0
    
    no_of_annotators_agreed = len(item.get('annotators', []))
    if no_of_annotators_agreed == 0:
        return 0.0
    ratio = no_of_annotators_agreed / max_annotators
    return ratio


def score_json_file(json_file: str, out_dir: str, file_name: str) -> None:
    """
    Process a single vote JSON file:
    - Group spans by PII type 
    - Compute triplet-based accuracies & posteriors per type
    - Attach probabilities
    - Write scored file to out_dir
    """

    # 1) Load file 
    with open(json_file, 'r') as f:
        items = json.load(f)

    # 2) Determine number of annotators
    all_anns = {ann for rec in items for ann in rec.get('annotators', [])}
    m = max(all_anns) + 1 if all_anns else 0

    print(f"Processing {json_file} with {m} annotators")

    # 3) Majority vote for each item
    for item in items:
        ratio = majority_vote(item, m)
        item['probability'] = ratio
        # Here we change to absolute bbox
        if item.get('bbox'):
            original_width = dimensions_map[file_name]['original_width']
            original_height = dimensions_map[file_name]['original_height']
            x = item['bbox'][0] * original_width
            y = item['bbox'][1] * original_height
            width = item['bbox'][2] * original_width
            height = item['bbox'][3] * original_height
        else:
            x = y = width = height = 0

        x0 = round(x)
        y0 = round(y)
        x1 = round(width)
        y1 = round(height)
        item['bbox'] = [x0, y0, x1, y1]

    # 4) Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # 5) Write scored file to out_dir
    filename = os.path.basename(json_file)
    output_path = os.path.join(out_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(items, f, indent=4)
    
    print(f"Wrote scored file: {output_path}")


def score_all_jsons_global(votes_dir: str, out_dir: str, dimensions_json_path: str | None = None) -> None:
    """
    Process all votes_*.json files in a directory:
    - Process each file independently
    - Write scored files to out_dir
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Process each vote file
    for fname in sorted(os.listdir(votes_dir)):
        if not (fname.startswith('votes_') and fname.endswith('.json')):
            continue
            
        input_path = os.path.join(votes_dir, fname)
        
        print(f"Processing: {fname}")
        
        # Process the file
        score_json_file(input_path, out_dir, file_name=fname)
    
    print(f"All files processed. Results saved to: {out_dir}")
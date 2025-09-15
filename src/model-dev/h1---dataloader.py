from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json
from qwen_vl_utils import process_vision_info
import torch
import os
import textwrap

class PIIVisionDataset(Dataset):
    def __init__(self, data_file, img_dir,  start_idx=0, end_idx=None):
        self.data = json.load(open(data_file))
        # self.data = self.data[:100]
        self.data = self.data[start_idx:end_idx]
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        try:
            image_file_path = os.path.join(self.img_dir, self.data[idx]['file_name'] + ".png")
            prompt = textwrap.dedent("""
                    Extract the following entity types from the image and return them in JSON format:

                    - Person Name  
                    - Email Address  
                    - Phone Number  
                    - Location  
                    - Organization Name  
                    - Date  
                    - Contract Number
                    - Invoice Number

                    For each extracted entity, include:
                    - "entity_type": The category of the extracted entity  
                    - "value": The exact text as it appears in the image  
                    - "bbox": The bounding box coordinates in the format [x_min, y_min, x_max, y_max] using pixel values  

                    Wrap the entire JSON output between <json-start> and <json-end> tags. 
                    Example:
                    <json-start>
                    [
                        {
                            "entity_type": "<extracted_entity_type>",
                            "value": "<extracted_text>",
                            "bbox": [<x_min>, <y_min>, <x_max>, <y_max>]
                        },
                        {
                            "entity_type": "<extracted_entity_type>",
                            "value": "<extracted_text>",
                            "bbox": [<x_min>, <y_min>, <x_max>, <y_max>]
                        }
                    ]
                    <json-end>
                    Return only the JSON object. Do not include any explanation or surrounding text.
                    """)
                
            labels = "<json-start>" + str(self.data[idx]['labels']) + "<json-end>"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type":"image", "image": image_file_path, "resized_width": 840, "resized_height": 840},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": labels},
                    ]
                }
            ]

            img_inputs, vid_inputs = process_vision_info(messages)
                
            messages[0]['content'][1]['image'] = img_inputs[0]

            return {"messages": messages, "file_name": self.data[idx]['file_name'], "labels": self.data[idx]['labels']}
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return None
        
        
        
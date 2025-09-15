import streamlit as st
import json
import os

st.title("OCR Text and Ground Truth vs Predictions Viewer")

uploaded = st.file_uploader("Upload JSON listing file", type=["json"])

def load_listing(file):
    try:
        return json.load(file)
    except Exception as e:
        st.error(f"Error loading JSON listing: {e}")
        return None

if uploaded:
    listing = load_listing(uploaded)
    if listing:
        # Create a mapping of index to display name
        options = []
        for idx, entry in enumerate(listing):
            # Use a snippet of OCR text or file name as label
            snippet = entry.get("ocr_text", "").split("\n")[0][:50]
            label = f"{idx}: {os.path.basename(entry.get('ground_truth_file_path', ''))} - {snippet}"
            options.append(label)
        selection = st.selectbox("Select an entry to view", options)
        idx = int(selection.split(":")[0])
        entry = listing[idx]

        st.subheader("OCR Text")
        st.text_area("OCR Text", entry.get("ocr_text", ""), height=200)

        col1, col2 = st.columns(2)
        # Load ground truth JSON
        gt_path = entry.get("ground_truth_file_path")
        pred_path = entry.get("predictions_file_path")
        with col1:
            st.subheader("Ground Truth JSON")
            if gt_path and os.path.exists(gt_path):
                try:
                    with open(gt_path, 'r', encoding='utf-8') as f:
                        gt_data = json.load(f)
                    st.json(gt_data)
                except Exception as e:
                    st.error(f"Error loading ground truth JSON: {e}")
            else:
                st.warning(f"Ground truth file not found at {gt_path}")

        with col2:
            st.subheader("Predictions JSON")
            if True:
                st.json(pred_path)
            # if pred_path and os.path.exists(pred_path):
            #     try:
            #         with open(pred_path, 'r', encoding='utf-8') as f:
            #             pred_data = json.load(f)
            #         st.json(pred_data)
            #     except Exception as e:
            #         st.error(f"Error loading predictions JSON: {e}")
            else:
                st.warning(f"Predictions file not found at {pred_path}")

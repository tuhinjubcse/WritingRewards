import streamlit as st
import json
import pandas as pd
import os
from typing import Dict, List

def load_data(eval_fn: str) -> tuple[List[Dict], Dict]:
    """Load evaluation data and create an id to data mapping"""
    with open(eval_fn) as f:
        data = json.load(f)
    
    id2data = {d["id"]: d for d in data}
    
    # Load predictions for each model
    for fn in os.listdir("data/preds"):
        with open(f"data/preds/{fn}") as f:
            model_name = fn.replace(".jsonl", "").replace("preds_", "")
            for line in f:
                d = json.loads(line)
                id2data[d["id"]][f"pred_{model_name}"] = d["output"]
    
    return data, id2data

def get_split_data(data: List[Dict], split: str) -> List[Dict]:
    """Filter data by split type"""
    split_map = {
        "Pairwise (P)": "pairwise",
        "Reward (R)": "reward", 
        "Gold (G)": "pairwise-gold",
        "Silver (S)": "pairwise-silver"
    }
    return [d for d in data if d["sample_type"] == split_map[split]]

def format_prediction(pred_dict: Dict) -> str:
    """Format prediction dictionary as readable string"""
    if "preference" in pred_dict:
        return f"Preference: {pred_dict['preference']}"
    elif "score" in pred_dict:
        return f"Score: {pred_dict['score']}"
    return str(pred_dict)

def main():
    st.title("Model Predictions Viewer")
    
    # Load data
    data, id2data = load_data("data/finetune_PRGS_test.json")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Split selection
    split_options = ["Pairwise (P)", "Reward (R)", "Gold (G)", "Silver (S)"]
    selected_split = st.sidebar.selectbox(
        "Select Data Split",
        split_options
    )
    
    # Filter data by split
    filtered_data = get_split_data(data, selected_split)
    
    # Get available models
    models = [
        key.replace("pred_", "") 
        for key in id2data[filtered_data[0]["id"]].keys() 
        if key.startswith("pred_")
    ]
    
    # Model selection - now defaulting to all models
    default_models = [m for m in models if not ("gem-1p5" in m and (m.endswith("b") or m.endswith("c")))]
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        models,
        default=default_models  # Changed to select all models by default
    )
    
    # Main content
    st.subheader(f"Showing {len(filtered_data)} samples from {selected_split} split")
    
    # Sample navigation
    sample_index = st.number_input(
        "Sample Index", 
        min_value=0,
        max_value=len(filtered_data)-1,
        value=0
    )
    
    if sample_index < len(filtered_data):
        sample = filtered_data[sample_index]
        
        # Display input text
        st.text_area("Input Text", sample["text_input"], height=300)
        
        # Display ground truth
        st.subheader("Ground Truth")
        if "reference_preference" in sample:
            st.write(f"Reference Preference: {sample['reference_preference']}")
        if "zscore" in sample:
            st.write(f"Z-Score: {sample['zscore']}")
            
        # Display model predictions
        st.subheader("Model Predictions")
        
        # Calculate number of rows needed
        MAX_COLS = 3
        num_models = len(selected_models)
        num_rows = (num_models + MAX_COLS - 1) // MAX_COLS  # Ceiling division
        
        # Display predictions in rows of 3
        for row in range(num_rows):
            start_idx = row * MAX_COLS
            end_idx = min(start_idx + MAX_COLS, num_models)
            row_models = selected_models[start_idx:end_idx]
            
            # Create columns for this row
            cols = st.columns(MAX_COLS)
            
            # Fill columns with model predictions
            for col_idx, model in enumerate(row_models):
                with cols[col_idx]:
                    st.write(f"**{model}**")
                    pred_key = f"pred_{model}"
                    if pred_key in sample:
                        st.write(format_prediction(sample[pred_key]))
                    else:
                        st.write("No prediction available")
            
            # Add some vertical spacing between rows
            st.write("")

if __name__ == "__main__":
    main() 

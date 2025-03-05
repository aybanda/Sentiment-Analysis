import os
import pandas as pd
from typing import Tuple
import numpy as np
from datetime import datetime
import streamlit as st

def create_sample_dataset(input_file: str, output_file: str, sample_size: int = 100000):
    """Create a smaller sample dataset"""
    try:
        df = pd.read_csv(input_file, encoding='latin-1', names=['target', 'id', 'date', 'flag', 'user', 'text'])
        # Ensure balanced sampling
        pos_samples = df[df['target'] == 4].sample(n=sample_size//2, random_state=42)
        neg_samples = df[df['target'] == 0].sample(n=sample_size//2, random_state=42)
        sampled_df = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42)
        sampled_df.to_csv(output_file, index=False)
        return True
    except Exception as e:
        print(f"Error creating sample: {str(e)}")
        return False

def load_sentiment140(split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split Sentiment140 dataset for initial training and continuous learning
    
    Args:
        split_ratio: Ratio of data to use for initial training
    
    Returns:
        Tuple of (training_data, continuous_learning_data)
    """
    try:
        # First try to load from local path
        local_path = 'data/sentiment140.csv'
        sample_path = 'data/sentiment140_sample.csv'
        
        if os.path.exists(local_path) and not os.path.exists(sample_path):
            st.info("Creating a smaller sample dataset for deployment...")
            if create_sample_dataset(local_path, sample_path):
                st.success("Sample dataset created successfully!")
            else:
                st.error("Failed to create sample dataset.")
        
        # Try to load the sample dataset first
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path, encoding='latin-1')
        elif os.path.exists(local_path):
            df = pd.read_csv(local_path, encoding='latin-1', names=['target', 'id', 'date', 'flag', 'user', 'text'])
        else:
            # If no local file exists, show file uploader in Streamlit
            st.warning("Dataset not found locally. Please upload the Sentiment140 dataset (full or sample).")
            uploaded_file = st.file_uploader("Upload sentiment140.csv", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
                # If the uploaded file is too large, sample it
                if len(df) > 100000:
                    st.info("Creating a smaller sample from the uploaded dataset...")
                    pos_samples = df[df['target'] == 4].sample(n=50000, random_state=42)
                    neg_samples = df[df['target'] == 0].sample(n=50000, random_state=42)
                    df = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42)
            else:
                return None, None

        # Process the dataframe
        df = df[['target', 'text']]  # Keep only target and text columns
        df['target'] = df['target'].map({0: 0, 4: 1})  # Map targets to binary
        train_size = int(len(df) * split_ratio)
        train_data = df[:train_size]
        test_data = df[train_size:]
        
        return train_data, test_data
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None

def get_batch_for_continuous_learning(data: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Get a random batch of examples for continuous learning"""
    return data.sample(n=min(batch_size, len(data))) 
import os
import pandas as pd
from typing import Tuple
import numpy as np
from datetime import datetime
import streamlit as st

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
        if os.path.exists(local_path):
            df = pd.read_csv(local_path, encoding='latin-1', names=['target', 'id', 'date', 'flag', 'user', 'text'])
        else:
            # If local file doesn't exist, show file uploader in Streamlit
            st.warning("Dataset not found locally. Please upload the Sentiment140 dataset.")
            uploaded_file = st.file_uploader("Upload sentiment140.csv", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, encoding='latin-1', names=['target', 'id', 'date', 'flag', 'user', 'text'])
            else:
                return None, None

        # Process the dataframe
        df = df[['target', 'text']]  # Keep only target and text columns
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
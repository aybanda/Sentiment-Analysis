import os
import pandas as pd
from typing import Tuple
import numpy as np
from datetime import datetime

def load_sentiment140(split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split Sentiment140 dataset for initial training and continuous learning
    
    Args:
        split_ratio: Ratio of data to use for initial training
    
    Returns:
        Tuple of (training_data, continuous_learning_data)
    """
    # Define column names
    columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Get the absolute path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'sentiment140.csv')
    
    # Load the dataset
    df = pd.read_csv(data_path, encoding='latin-1', names=columns)
    
    # Convert labels from 0/4 to 0/1
    df['target'] = df['target'].map({0: 0, 4: 1})
    
    # Convert dates ignoring timezone
    df['date'] = pd.to_datetime(df['date'].str.replace('PDT', '').str.replace('PST', ''), 
                               format='%a %b %d %H:%M:%S %Y')
    df = df.sort_values('date')
    
    # Split data
    split_idx = int(len(df) * split_ratio)
    train_data = df.iloc[:split_idx][['text', 'target']]
    continuous_data = df.iloc[split_idx:][['text', 'target']]
    
    return train_data, continuous_data

def get_batch_for_continuous_learning(data: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Get a random batch of examples for continuous learning"""
    return data.sample(n=min(batch_size, len(data))) 
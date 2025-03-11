#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the emotion recognition project.
This module contains common functions used across different components of the project.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def create_directory(directory_path):
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create
    """
    os.makedirs(directory_path, exist_ok=True)

def save_dataframe(df, file_path, index=False):
    """
    Save a DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the DataFrame
        index: Whether to include the index in the CSV file
    """
    df.to_csv(file_path, index=index)
    print(f"DataFrame saved to {file_path}")

def load_dataframe(file_path):
    """
    Load a DataFrame from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame loaded from the CSV file
    """
    df = pd.read_csv(file_path)
    print(f"DataFrame loaded from {file_path}")
    return df

def normalize_vad_values(valence, arousal, dominance):
    """
    Normalize VAD values from 1-5 scale to 0-1 scale.
    
    Args:
        valence: Valence value (1-5)
        arousal: Arousal/Activation value (1-5)
        dominance: Dominance value (1-5)
        
    Returns:
        Normalized VAD values (0-1)
    """
    valence_norm = (valence - 1) / 4
    arousal_norm = (arousal - 1) / 4
    dominance_norm = (dominance - 1) / 4
    
    return valence_norm, arousal_norm, dominance_norm

def denormalize_vad_values(valence_norm, arousal_norm, dominance_norm):
    """
    Denormalize VAD values from 0-1 scale to 1-5 scale.
    
    Args:
        valence_norm: Normalized valence value (0-1)
        arousal_norm: Normalized arousal/activation value (0-1)
        dominance_norm: Normalized dominance value (0-1)
        
    Returns:
        Denormalized VAD values (1-5)
    """
    valence = (valence_norm * 4) + 1
    arousal = (arousal_norm * 4) + 1
    dominance = (dominance_norm * 4) + 1
    
    return valence, arousal, dominance

def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix', output_file=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        title: Title of the plot
        output_file: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Confusion matrix saved to {output_file}")
    
    plt.close()

def print_classification_metrics(y_true, y_pred, labels=None):
    """
    Print classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
    """
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels))

def get_project_root():
    """
    Get the root directory of the project.
    
    Returns:
        Absolute path to the project root directory
    """
    # This assumes the utils.py file is in the project root directory
    return os.path.dirname(os.path.abspath(__file__))

def get_data_directory():
    """
    Get the data directory of the project.
    
    Returns:
        Absolute path to the data directory
    """
    return os.path.join(get_project_root(), 'data')

def get_raw_data_directory():
    """
    Get the raw data directory of the project.
    
    Returns:
        Absolute path to the raw data directory
    """
    return os.path.join(get_data_directory(), 'raw')

def get_processed_data_directory():
    """
    Get the processed data directory of the project.
    
    Returns:
        Absolute path to the processed data directory
    """
    return os.path.join(get_data_directory(), 'processed')

def get_models_directory():
    """
    Get the models directory of the project.
    
    Returns:
        Absolute path to the models directory
    """
    return os.path.join(get_project_root(), 'models')

def get_results_directory():
    """
    Get the results directory of the project.
    
    Returns:
        Absolute path to the results directory
    """
    return os.path.join(get_project_root(), 'results')

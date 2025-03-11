#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preprocessing script for IEMOCAP dataset.
This script extracts VAD (Valence-Arousal-Dominance) annotations from attribute files.
"""

import os
import re
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def extract_vad_from_file(file_path):
    """
    Extract VAD values from attribute files.
    
    Args:
        file_path: Path to the attribute file
        
    Returns:
        DataFrame with utterance IDs and their VAD values
    """
    data = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # Extract utterance ID and VAD values using regex
        match = re.match(r'(\S+)\s+:act\s+(\d+);\s+:val\s+(\d+);\s+:dom\s+(\d+);', line)
        if match:
            utterance_id = match.group(1)
            activation = int(match.group(2))
            valence = int(match.group(3))
            dominance = int(match.group(4))
            
            # Normalize to [0, 1] range (original scale is 1-5)
            activation_norm = (activation - 1) / 4
            valence_norm = (valence - 1) / 4
            dominance_norm = (dominance - 1) / 4
            
            data.append({
                'utterance_id': utterance_id,
                'activation': activation,
                'valence': valence,
                'dominance': dominance,
                'activation_norm': activation_norm,
                'valence_norm': valence_norm,
                'dominance_norm': dominance_norm,
                'source_file': os.path.basename(file_path)
            })
    
    return pd.DataFrame(data)

def process_all_attribute_files(data_dir):
    """
    Process all attribute files in the dataset.
    
    Args:
        data_dir: Root directory of the dataset
        
    Returns:
        DataFrame with all VAD annotations
    """
    # Find all attribute files
    attr_files = glob.glob(os.path.join(data_dir, '**/*_atr.txt'), recursive=True)
    
    all_data = []
    for file_path in attr_files:
        df = extract_vad_from_file(file_path)
        all_data.append(df)
    
    # Combine all data
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def map_vad_to_emotion(vad_df, method='quadrant'):
    """
    Map VAD values to emotion categories.
    
    Args:
        vad_df: DataFrame with VAD values
        method: Method to use for mapping ('quadrant' or 'custom')
        
    Returns:
        DataFrame with emotion labels added
    """
    df = vad_df.copy()
    
    if method == 'quadrant':
        # Simple quadrant-based mapping using valence and activation
        # High valence, high arousal -> Happy/Excited
        # High valence, low arousal -> Calm/Content
        # Low valence, high arousal -> Angry/Frustrated
        # Low valence, low arousal -> Sad/Bored
        
        # Define midpoint (0.5 for normalized values)
        midpoint = 0.5
        
        def assign_emotion(row):
            v = row['valence_norm']
            a = row['activation_norm']
            
            if v >= midpoint and a >= midpoint:
                return 'happy'
            elif v >= midpoint and a < midpoint:
                return 'calm'
            elif v < midpoint and a >= midpoint:
                return 'angry'
            else:
                return 'sad'
        
        df['emotion'] = df.apply(assign_emotion, axis=1)
    
    elif method == 'custom':
        # More nuanced mapping based on all three dimensions
        # This is a placeholder for a more sophisticated mapping
        # that could be implemented based on literature
        
        def assign_emotion_custom(row):
            v = row['valence_norm']
            a = row['activation_norm']
            d = row['dominance_norm']
            
            # These thresholds can be adjusted based on literature
            if v >= 0.7 and a >= 0.7:
                return 'excited'
            elif v >= 0.7 and a < 0.3:
                return 'content'
            elif v < 0.3 and a >= 0.7 and d >= 0.7:
                return 'angry'
            elif v < 0.3 and a >= 0.7 and d < 0.3:
                return 'afraid'
            elif v < 0.3 and a < 0.3:
                return 'sad'
            elif v >= 0.5 and a >= 0.4 and a <= 0.6:
                return 'happy'
            else:
                return 'neutral'
        
        df['emotion'] = df.apply(assign_emotion_custom, axis=1)
    
    return df

def main():
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing attribute files from {data_dir}...")
    
    # Process all attribute files
    vad_df = process_all_attribute_files(data_dir)
    
    if vad_df.empty:
        print("No attribute files found or processed.")
        return
    
    # Save the raw VAD values
    vad_output_path = os.path.join(output_dir, 'vad_annotations.csv')
    vad_df.to_csv(vad_output_path, index=False)
    print(f"Saved VAD annotations to {vad_output_path}")
    
    # Map VAD to emotions using quadrant method
    emotion_df_quadrant = map_vad_to_emotion(vad_df, method='quadrant')
    quadrant_output_path = os.path.join(output_dir, 'emotion_quadrant.csv')
    emotion_df_quadrant.to_csv(quadrant_output_path, index=False)
    print(f"Saved quadrant-based emotion mapping to {quadrant_output_path}")
    
    # Map VAD to emotions using custom method
    emotion_df_custom = map_vad_to_emotion(vad_df, method='custom')
    custom_output_path = os.path.join(output_dir, 'emotion_custom.csv')
    emotion_df_custom.to_csv(custom_output_path, index=False)
    print(f"Saved custom emotion mapping to {custom_output_path}")
    
    # Print some statistics
    print("\nDataset statistics:")
    print(f"Total number of utterances: {len(vad_df)}")
    
    # Distribution of VAD values
    print("\nVAD value distributions (original scale 1-5):")
    for dim in ['activation', 'valence', 'dominance']:
        print(f"{dim.capitalize()} distribution:")
        print(vad_df[dim].value_counts().sort_index())
    
    # Distribution of emotions
    print("\nEmotion distribution (quadrant method):")
    print(emotion_df_quadrant['emotion'].value_counts())
    
    print("\nEmotion distribution (custom method):")
    print(emotion_df_custom['emotion'].value_counts())

if __name__ == "__main__":
    main()

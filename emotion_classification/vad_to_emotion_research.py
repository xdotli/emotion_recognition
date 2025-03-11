#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Research on VAD to emotion mapping approaches.
This script implements various methods for mapping VAD (Valence-Arousal-Dominance) values to emotion categories.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def plot_vad_space(vad_df, emotion_column='emotion', output_file='vad_space.png'):
    """
    Plot the VAD space with emotions color-coded.
    
    Args:
        vad_df: DataFrame with VAD values and emotion labels
        emotion_column: Column name for emotion labels
        output_file: Path to save the output plot
    """
    plt.figure(figsize=(15, 12))
    
    # 3D plot
    ax1 = plt.subplot(221, projection='3d')
    emotions = vad_df[emotion_column].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(emotions)))
    
    for i, emotion in enumerate(emotions):
        subset = vad_df[vad_df[emotion_column] == emotion]
        ax1.scatter(
            subset['valence_norm'], 
            subset['activation_norm'], 
            subset['dominance_norm'],
            c=[colors[i]],
            label=emotion,
            alpha=0.7
        )
    
    ax1.set_xlabel('Valence')
    ax1.set_ylabel('Activation')
    ax1.set_zlabel('Dominance')
    ax1.set_title('3D VAD Space')
    ax1.legend()
    
    # 2D plot: Valence-Arousal
    ax2 = plt.subplot(222)
    for i, emotion in enumerate(emotions):
        subset = vad_df[vad_df[emotion_column] == emotion]
        ax2.scatter(
            subset['valence_norm'], 
            subset['activation_norm'],
            c=[colors[i]],
            label=emotion,
            alpha=0.7
        )
    
    ax2.set_xlabel('Valence')
    ax2.set_ylabel('Activation')
    ax2.set_title('Valence-Activation Space')
    ax2.grid(True)
    
    # 2D plot: Valence-Dominance
    ax3 = plt.subplot(223)
    for i, emotion in enumerate(emotions):
        subset = vad_df[vad_df[emotion_column] == emotion]
        ax3.scatter(
            subset['valence_norm'], 
            subset['dominance_norm'],
            c=[colors[i]],
            label=emotion,
            alpha=0.7
        )
    
    ax3.set_xlabel('Valence')
    ax3.set_ylabel('Dominance')
    ax3.set_title('Valence-Dominance Space')
    ax3.grid(True)
    
    # 2D plot: Arousal-Dominance
    ax4 = plt.subplot(224)
    for i, emotion in enumerate(emotions):
        subset = vad_df[vad_df[emotion_column] == emotion]
        ax4.scatter(
            subset['activation_norm'], 
            subset['dominance_norm'],
            c=[colors[i]],
            label=emotion,
            alpha=0.7
        )
    
    ax4.set_xlabel('Activation')
    ax4.set_ylabel('Dominance')
    ax4.set_title('Activation-Dominance Space')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"VAD space plot saved to {output_file}")

def quadrant_mapping(vad_df):
    """
    Map VAD values to emotions using the quadrant method.
    
    Args:
        vad_df: DataFrame with VAD values
        
    Returns:
        DataFrame with emotion labels added
    """
    df = vad_df.copy()
    
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
    
    df['emotion_quadrant'] = df.apply(assign_emotion, axis=1)
    
    return df

def custom_mapping(vad_df):
    """
    Map VAD values to emotions using a custom method that considers all three dimensions.
    
    Args:
        vad_df: DataFrame with VAD values
        
    Returns:
        DataFrame with emotion labels added
    """
    df = vad_df.copy()
    
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
    
    df['emotion_custom'] = df.apply(assign_emotion_custom, axis=1)
    
    return df

def plutchik_mapping(vad_df):
    """
    Map VAD values to emotions using Plutchik's wheel of emotions.
    
    Args:
        vad_df: DataFrame with VAD values
        
    Returns:
        DataFrame with emotion labels added
    """
    df = vad_df.copy()
    
    def assign_emotion_plutchik(row):
        v = row['valence_norm']
        a = row['activation_norm']
        d = row['dominance_norm']
        
        # Primary emotions based on Plutchik's wheel
        if v >= 0.7 and a >= 0.7 and d >= 0.7:
            return 'joy'
        elif v >= 0.6 and a >= 0.7 and d < 0.4:
            return 'trust'
        elif v >= 0.6 and a < 0.4 and d >= 0.6:
            return 'anticipation'
        elif v < 0.4 and a >= 0.7 and d >= 0.7:
            return 'anger'
        elif v < 0.3 and a >= 0.6 and d < 0.4:
            return 'fear'
        elif v < 0.3 and a < 0.4 and d < 0.3:
            return 'sadness'
        elif v < 0.4 and a < 0.3 and d >= 0.6:
            return 'disgust'
        elif v >= 0.6 and a < 0.3 and d < 0.4:
            return 'surprise'
        else:
            return 'neutral'
    
    df['emotion_plutchik'] = df.apply(assign_emotion_plutchik, axis=1)
    
    return df

def ekman_mapping(vad_df):
    """
    Map VAD values to Ekman's six basic emotions.
    
    Args:
        vad_df: DataFrame with VAD values
        
    Returns:
        DataFrame with emotion labels added
    """
    df = vad_df.copy()
    
    def assign_emotion_ekman(row):
        v = row['valence_norm']
        a = row['activation_norm']
        d = row['dominance_norm']
        
        # Ekman's six basic emotions
        if v >= 0.7 and a >= 0.5:
            return 'happiness'
        elif v < 0.3 and a >= 0.7 and d >= 0.6:
            return 'anger'
        elif v < 0.3 and a >= 0.7 and d < 0.4:
            return 'fear'
        elif v < 0.3 and a < 0.4:
            return 'sadness'
        elif v < 0.4 and a < 0.3 and d >= 0.6:
            return 'disgust'
        elif v >= 0.5 and a >= 0.7 and d < 0.5:
            return 'surprise'
        else:
            return 'neutral'
    
    df['emotion_ekman'] = df.apply(assign_emotion_ekman, axis=1)
    
    return df

def compare_mappings(vad_df):
    """
    Compare different VAD to emotion mapping methods.
    
    Args:
        vad_df: DataFrame with VAD values
        
    Returns:
        DataFrame with emotion labels from different methods
    """
    df = vad_df.copy()
    
    # Apply different mapping methods
    df = quadrant_mapping(df)
    df = custom_mapping(df)
    df = plutchik_mapping(df)
    df = ekman_mapping(df)
    
    # Count emotions for each method
    print("Emotion distribution (Quadrant method):")
    print(df['emotion_quadrant'].value_counts())
    
    print("\nEmotion distribution (Custom method):")
    print(df['emotion_custom'].value_counts())
    
    print("\nEmotion distribution (Plutchik's wheel):")
    print(df['emotion_plutchik'].value_counts())
    
    print("\nEmotion distribution (Ekman's basic emotions):")
    print(df['emotion_ekman'].value_counts())
    
    return df

def plot_emotion_distributions(df, output_file='emotion_distributions.png'):
    """
    Plot emotion distributions for different mapping methods.
    
    Args:
        df: DataFrame with emotion labels from different methods
        output_file: Path to save the output plot
    """
    plt.figure(figsize=(20, 15))
    
    # Quadrant method
    plt.subplot(221)
    sns.countplot(x='emotion_quadrant', data=df)
    plt.title('Quadrant Method')
    plt.xticks(rotation=45)
    
    # Custom method
    plt.subplot(222)
    sns.countplot(x='emotion_custom', data=df)
    plt.title('Custom Method')
    plt.xticks(rotation=45)
    
    # Plutchik's wheel
    plt.subplot(223)
    sns.countplot(x='emotion_plutchik', data=df)
    plt.title('Plutchik\'s Wheel')
    plt.xticks(rotation=45)
    
    # Ekman's basic emotions
    plt.subplot(224)
    sns.countplot(x='emotion_ekman', data=df)
    plt.title('Ekman\'s Basic Emotions')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Emotion distribution plot saved to {output_file}")

def main():
    """
    Main function to research VAD to emotion mapping approaches.
    """
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'emotion_classification')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAD annotations
    vad_df = pd.read_csv(os.path.join(processed_dir, 'vad_annotations.csv'))
    
    print("Researching VAD to emotion mapping approaches...")
    print(f"Number of utterances: {len(vad_df)}")
    
    # Compare different mapping methods
    df_with_emotions = compare_mappings(vad_df)
    
    # Plot VAD space with emotions
    plot_vad_space(df_with_emotions, emotion_column='emotion_quadrant', 
                  output_file=os.path.join(output_dir, 'vad_space_quadrant.png'))
    
    plot_vad_space(df_with_emotions, emotion_column='emotion_custom', 
                  output_file=os.path.join(output_dir, 'vad_space_custom.png'))
    
    plot_vad_space(df_with_emotions, emotion_column='emotion_plutchik', 
                  output_file=os.path.join(output_dir, 'vad_space_plutchik.png'))
    
    plot_vad_space(df_with_emotions, emotion_column='emotion_ekman', 
                  output_file=os.path.join(output_dir, 'vad_space_ekman.png'))
    
    # Plot emotion distributions
    plot_emotion_distributions(df_with_emotions, 
                              output_file=os.path.join(output_dir, 'emotion_distributions.png'))
    
    # Save the DataFrame with emotion labels
    df_with_emotions.to_csv(os.path.join(output_dir, 'vad_with_emotions.csv'), index=False)
    print(f"DataFrame with emotion labels saved to {os.path.join(output_dir, 'vad_with_emotions.csv')}")
    
    print("\nResearch on VAD to emotion mapping approaches completed.")
    print("Summary of findings:")
    print("1. Quadrant method: Simple but effective, divides the VA space into four quadrants")
    print("2. Custom method: More nuanced, considers all three dimensions")
    print("3. Plutchik's wheel: Based on psychological theory, eight primary emotions")
    print("4. Ekman's basic emotions: Six universal emotions recognized across cultures")
    print("\nRecommendation: Use Plutchik's or Ekman's approach for a psychologically grounded mapping")

if __name__ == "__main__":
    main()

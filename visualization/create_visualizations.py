#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization script for emotion recognition results.
This script creates various visualizations to better understand the emotion distribution and VAD space.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import joblib
import json

def plot_emotion_distribution(df, output_dir):
    """
    Plot emotion distribution for different emotion models.
    
    Args:
        df: DataFrame with emotion labels
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get emotion columns
    emotion_columns = [col for col in df.columns if col.startswith('emotion_')]
    
    # Plot distribution for each emotion model
    for emotion_col in emotion_columns:
        plt.figure(figsize=(12, 8))
        
        # Count emotions
        emotion_counts = df[emotion_col].value_counts()
        
        # Plot
        ax = sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
        
        # Add value labels on top of bars
        for i, count in enumerate(emotion_counts.values):
            ax.text(i, count + 5, str(count), ha='center')
        
        # Set title and labels
        plt.title(f'Emotion Distribution - {emotion_col.replace("emotion_", "").capitalize()} Model')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'emotion_distribution_{emotion_col.replace("emotion_", "")}.png'))
        plt.close()
        
        print(f"Emotion distribution plot saved for {emotion_col}")

def plot_vad_3d_scatter(df, output_dir):
    """
    Create 3D scatter plot of VAD values colored by emotion.
    
    Args:
        df: DataFrame with VAD values and emotion labels
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get emotion columns
    emotion_columns = [col for col in df.columns if col.startswith('emotion_')]
    
    # Plot 3D scatter for each emotion model
    for emotion_col in emotion_columns:
        plt.figure(figsize=(12, 10))
        ax = plt.axes(projection='3d')
        
        # Get unique emotions
        emotions = df[emotion_col].unique()
        
        # Create colormap
        cmap = plt.cm.get_cmap('tab10', len(emotions))
        
        # Plot each emotion
        for i, emotion in enumerate(emotions):
            subset = df[df[emotion_col] == emotion]
            ax.scatter(
                subset['valence_norm'],
                subset['activation_norm'],
                subset['dominance_norm'],
                c=[cmap(i)],
                label=emotion,
                alpha=0.7,
                s=50
            )
        
        # Set labels and title
        ax.set_xlabel('Valence')
        ax.set_ylabel('Activation')
        ax.set_zlabel('Dominance')
        ax.set_title(f'3D VAD Space - {emotion_col.replace("emotion_", "").capitalize()} Model')
        
        # Add legend
        ax.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'vad_3d_scatter_{emotion_col.replace("emotion_", "")}.png'))
        plt.close()
        
        print(f"3D VAD scatter plot saved for {emotion_col}")

def plot_vad_pairplot(df, output_dir):
    """
    Create pairplot of VAD values colored by emotion.
    
    Args:
        df: DataFrame with VAD values and emotion labels
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get emotion columns
    emotion_columns = [col for col in df.columns if col.startswith('emotion_')]
    
    # Plot pairplot for each emotion model
    for emotion_col in emotion_columns:
        # Create subset with VAD values and emotion
        subset = df[['valence_norm', 'activation_norm', 'dominance_norm', emotion_col]].copy()
        
        # Rename columns for better readability
        subset.columns = ['Valence', 'Activation', 'Dominance', 'Emotion']
        
        # Create pairplot
        g = sns.pairplot(subset, hue='Emotion', palette='tab10', height=3)
        
        # Set title
        g.fig.suptitle(f'VAD Pairplot - {emotion_col.replace("emotion_", "").capitalize()} Model', y=1.02)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'vad_pairplot_{emotion_col.replace("emotion_", "")}.png'))
        plt.close()
        
        print(f"VAD pairplot saved for {emotion_col}")

def plot_vad_heatmap(df, output_dir):
    """
    Create heatmap of VAD values.
    
    Args:
        df: DataFrame with VAD values
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate correlation matrix
    corr = df[['valence_norm', 'activation_norm', 'dominance_norm']].corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of VAD Dimensions')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'vad_correlation_heatmap.png'))
    plt.close()
    
    print("VAD correlation heatmap saved")

def plot_vad_dimension_histograms(df, output_dir):
    """
    Create histograms of VAD dimensions.
    
    Args:
        df: DataFrame with VAD values
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histograms
    plt.figure(figsize=(15, 5))
    
    # Valence
    plt.subplot(1, 3, 1)
    sns.histplot(df['valence_norm'], kde=True)
    plt.title('Valence Distribution')
    plt.xlabel('Valence')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Activation
    plt.subplot(1, 3, 2)
    sns.histplot(df['activation_norm'], kde=True)
    plt.title('Activation Distribution')
    plt.xlabel('Activation')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Dominance
    plt.subplot(1, 3, 3)
    sns.histplot(df['dominance_norm'], kde=True)
    plt.title('Dominance Distribution')
    plt.xlabel('Dominance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'vad_dimension_histograms.png'))
    plt.close()
    
    print("VAD dimension histograms saved")

def plot_vad_dimension_boxplots_by_emotion(df, output_dir):
    """
    Create boxplots of VAD dimensions grouped by emotion.
    
    Args:
        df: DataFrame with VAD values and emotion labels
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get emotion columns
    emotion_columns = [col for col in df.columns if col.startswith('emotion_')]
    
    # Plot boxplots for each emotion model
    for emotion_col in emotion_columns:
        # Create figure
        plt.figure(figsize=(15, 15))
        
        # Valence
        plt.subplot(3, 1, 1)
        sns.boxplot(x=emotion_col, y='valence_norm', data=df)
        plt.title(f'Valence by Emotion - {emotion_col.replace("emotion_", "").capitalize()} Model')
        plt.xlabel('Emotion')
        plt.ylabel('Valence')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Activation
        plt.subplot(3, 1, 2)
        sns.boxplot(x=emotion_col, y='activation_norm', data=df)
        plt.title(f'Activation by Emotion - {emotion_col.replace("emotion_", "").capitalize()} Model')
        plt.xlabel('Emotion')
        plt.ylabel('Activation')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Dominance
        plt.subplot(3, 1, 3)
        sns.boxplot(x=emotion_col, y='dominance_norm', data=df)
        plt.title(f'Dominance by Emotion - {emotion_col.replace("emotion_", "").capitalize()} Model')
        plt.xlabel('Emotion')
        plt.ylabel('Dominance')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'vad_boxplots_{emotion_col.replace("emotion_", "")}.png'))
        plt.close()
        
        print(f"VAD boxplots saved for {emotion_col}")

def plot_tsne_visualization(df, output_dir):
    """
    Create t-SNE visualization of VAD values colored by emotion.
    
    Args:
        df: DataFrame with VAD values and emotion labels
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get VAD values
    X = df[['valence_norm', 'activation_norm', 'dominance_norm']].values
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Get emotion columns
    emotion_columns = [col for col in df.columns if col.startswith('emotion_')]
    
    # Plot t-SNE for each emotion model
    for emotion_col in emotion_columns:
        plt.figure(figsize=(12, 10))
        
        # Get unique emotions
        emotions = df[emotion_col].unique()
        
        # Create colormap
        cmap = plt.cm.get_cmap('tab10', len(emotions))
        
        # Plot each emotion
        for i, emotion in enumerate(emotions):
            subset = df[df[emotion_col] == emotion]
            subset_indices = subset.index
            plt.scatter(
                X_tsne[subset_indices, 0],
                X_tsne[subset_indices, 1],
                c=[cmap(i)],
                label=emotion,
                alpha=0.7,
                s=50
            )
        
        # Set labels and title
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title(f't-SNE Visualization - {emotion_col.replace("emotion_", "").capitalize()} Model')
        
        # Add legend
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'tsne_visualization_{emotion_col.replace("emotion_", "")}.png'))
        plt.close()
        
        print(f"t-SNE visualization saved for {emotion_col}")

def plot_pca_visualization(df, output_dir):
    """
    Create PCA visualization of VAD values colored by emotion.
    
    Args:
        df: DataFrame with VAD values and emotion labels
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get VAD values
    X = df[['valence_norm', 'activation_norm', 'dominance_norm']].values
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Get explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    
    # Get emotion columns
    emotion_columns = [col for col in df.columns if col.startswith('emotion_')]
    
    # Plot PCA for each emotion model
    for emotion_col in emotion_columns:
        plt.figure(figsize=(12, 10))
        
        # Get unique emotions
        emotions = df[emotion_col].unique()
        
        # Create colormap
        cmap = plt.cm.get_cmap('tab10', len(emotions))
        
        # Plot each emotion
        for i, emotion in enumerate(emotions):
            subset = df[df[emotion_col] == emotion]
            subset_indices = subset.index
            plt.scatter(
                X_pca[subset_indices, 0],
                X_pca[subset_indices, 1],
                c=[cmap(i)],
                label=emotion,
                alpha=0.7,
                s=50
            )
        
        # Set labels and title
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
        plt.title(f'PCA Visualization - {emotion_col.replace("emotion_", "").capitalize()} Model')
        
        # Add legend
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pca_visualization_{emotion_col.replace("emotion_", "")}.png'))
        plt.close()
        
        print(f"PCA visualization saved for {emotion_col}")

def main():
    """
    Main function to create visualizations of emotion recognition results.
    """
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, 'emotion_classification', 'vad_with_emotions.csv')
    output_dir = os.path.join(base_dir, 'visualization', 'results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating visualizations of emotion recognition results...")
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Create visualizations
    plot_emotion_distribution(df, output_dir)
    plot_vad_3d_scatter(df, output_dir)
    plot_vad_pairplot(df, output_dir)
    plot_vad_heatmap(df, output_dir)
    plot_vad_dimension_histograms(df, output_dir)
    plot_vad_dimension_boxplots_by_emotion(df, output_dir)
    plot_tsne_visualization(df, output_dir)
    plot_pca_visualization(df, output_dir)
    
    print("\nVisualizations completed.")

if __name__ == "__main__":
    main()

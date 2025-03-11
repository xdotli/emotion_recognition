#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model evaluation script for emotion recognition system.
This script evaluates the performance of the VAD-to-emotion classifiers.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotion_classification.vad_to_emotion_classifier import VADToEmotionClassifier

def load_results(models_dir):
    """
    Load classifier results.
    
    Args:
        models_dir: Directory containing classifier results
        
    Returns:
        DataFrame with classifier results
    """
    results_file = os.path.join(models_dir, 'classifier_results.csv')
    if os.path.exists(results_file):
        return pd.read_csv(results_file)
    else:
        return None

def plot_accuracy_comparison(results_df, output_file='accuracy_comparison.png'):
    """
    Plot accuracy comparison between different classifiers and emotion models.
    
    Args:
        results_df: DataFrame with classifier results
        output_file: Path to save the output plot
    """
    plt.figure(figsize=(12, 8))
    
    # Group by emotion model and classifier type
    grouped = results_df.groupby(['emotion_model', 'classifier_type'])['accuracy'].mean().reset_index()
    
    # Pivot for plotting
    pivot_df = grouped.pivot(index='emotion_model', columns='classifier_type', values='accuracy')
    
    # Plot
    ax = pivot_df.plot(kind='bar', width=0.8)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3)
    
    plt.title('Accuracy Comparison by Emotion Model and Classifier Type')
    plt.xlabel('Emotion Model')
    plt.ylabel('Accuracy')
    plt.ylim(0.95, 1.01)  # Adjust y-axis to better show differences
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Accuracy comparison plot saved to {output_file}")

def perform_cross_validation(model_dir, data_file, emotion_model):
    """
    Perform cross-validation on a trained model.
    
    Args:
        model_dir: Directory containing the model
        data_file: Path to the data file
        emotion_model: Emotion model name
        
    Returns:
        Cross-validation scores
    """
    # Load model
    classifier = VADToEmotionClassifier.load_model(model_dir)
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Prepare features and target
    X = df[['valence_norm', 'activation_norm', 'dominance_norm']]
    y = df[f'emotion_{emotion_model}']
    
    # Scale features
    X_scaled = classifier.scaler.transform(X)
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(classifier.classifier, X_scaled, y, cv=cv, scoring='accuracy')
    
    return scores

def evaluate_all_models(models_dir, data_file, output_dir):
    """
    Evaluate all trained models with cross-validation.
    
    Args:
        models_dir: Directory containing trained models
        data_file: Path to the data file
        output_dir: Directory to save evaluation results
    """
    # Load results
    results_df = load_results(models_dir)
    
    if results_df is None:
        print("No results file found.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy comparison
    plot_accuracy_comparison(results_df, os.path.join(output_dir, 'accuracy_comparison.png'))
    
    # Perform cross-validation for each model
    cv_results = []
    
    for _, row in results_df.iterrows():
        emotion_model = row['emotion_model']
        classifier_type = row['classifier_type']
        model_dir = os.path.join(models_dir, f"{emotion_model}_{classifier_type}")
        
        print(f"Performing cross-validation for {emotion_model.upper()} with {classifier_type.upper()}...")
        
        scores = perform_cross_validation(model_dir, data_file, emotion_model)
        
        cv_results.append({
            'emotion_model': emotion_model,
            'classifier_type': classifier_type,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_min': scores.min(),
            'cv_max': scores.max()
        })
    
    # Create DataFrame with cross-validation results
    cv_df = pd.DataFrame(cv_results)
    
    # Save cross-validation results
    cv_df.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)
    
    # Plot cross-validation results
    plt.figure(figsize=(12, 8))
    
    # Group by emotion model and classifier type
    grouped = cv_df.groupby(['emotion_model', 'classifier_type'])['cv_mean'].mean().reset_index()
    
    # Pivot for plotting
    pivot_df = grouped.pivot(index='emotion_model', columns='classifier_type', values='cv_mean')
    
    # Plot
    ax = pivot_df.plot(kind='bar', width=0.8, yerr=cv_df.groupby(['emotion_model', 'classifier_type'])['cv_std'].mean().reset_index().pivot(
        index='emotion_model', columns='classifier_type', values='cv_std'
    ))
    
    # Add value labels on top of bars - need to handle differently for error bars
    for i, emotion_model in enumerate(pivot_df.index):
        for j, classifier_type in enumerate(pivot_df.columns):
            value = pivot_df.iloc[i, j]
            ax.text(i + (j - 1) * 0.25, value + 0.005, f'{value:.4f}', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.title('Cross-Validation Accuracy by Emotion Model and Classifier Type')
    plt.xlabel('Emotion Model')
    plt.ylabel('Mean CV Accuracy')
    plt.ylim(0.95, 1.01)  # Adjust y-axis to better show differences
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_validation_comparison.png'))
    plt.close()
    
    print(f"Cross-validation comparison plot saved to {os.path.join(output_dir, 'cross_validation_comparison.png')}")
    
    # Print summary
    print("\nCross-validation results summary:")
    print(f"{'='*50}")
    for result in sorted(cv_results, key=lambda x: x['cv_mean'], reverse=True):
        print(f"Emotion model: {result['emotion_model'].upper()}, "
              f"Classifier: {result['classifier_type'].upper()}, "
              f"Mean CV Accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    
    # Find best model
    best_result = max(cv_results, key=lambda x: x['cv_mean'])
    print(f"\nBest model based on cross-validation: {best_result['emotion_model'].upper()} with "
          f"{best_result['classifier_type'].upper()} classifier "
          f"(Mean CV Accuracy: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f})")

def analyze_feature_importance(models_dir, output_dir):
    """
    Analyze feature importance for Random Forest classifiers.
    
    Args:
        models_dir: Directory containing trained models
        output_dir: Directory to save analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Random Forest models
    rf_models = []
    for root, dirs, files in os.walk(models_dir):
        if 'classifier.pkl' in files and 'random_forest' in root:
            rf_models.append(root)
    
    # Analyze feature importance for each model
    feature_importance = []
    
    for model_dir in rf_models:
        # Extract emotion model from directory name
        emotion_model = os.path.basename(model_dir).split('_')[0]
        
        # Load model
        classifier = VADToEmotionClassifier.load_model(model_dir)
        
        # Get feature importance
        importances = classifier.classifier.feature_importances_
        
        # Store results
        feature_importance.append({
            'emotion_model': emotion_model,
            'valence_importance': importances[0],
            'arousal_importance': importances[1],
            'dominance_importance': importances[2]
        })
    
    # Create DataFrame with feature importance
    fi_df = pd.DataFrame(feature_importance)
    
    # Save feature importance
    fi_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    
    # Melt DataFrame for plotting
    melted_df = pd.melt(fi_df, id_vars=['emotion_model'], 
                        value_vars=['valence_importance', 'arousal_importance', 'dominance_importance'],
                        var_name='feature', value_name='importance')
    
    # Clean feature names
    melted_df['feature'] = melted_df['feature'].str.replace('_importance', '')
    
    # Plot
    sns.barplot(x='emotion_model', y='importance', hue='feature', data=melted_df)
    
    plt.title('Feature Importance by Emotion Model (Random Forest)')
    plt.xlabel('Emotion Model')
    plt.ylabel('Feature Importance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    
    print(f"Feature importance plot saved to {os.path.join(output_dir, 'feature_importance.png')}")
    
    # Print summary
    print("\nFeature importance summary:")
    print(f"{'='*50}")
    for _, row in fi_df.iterrows():
        print(f"Emotion model: {row['emotion_model'].upper()}")
        print(f"  Valence importance: {row['valence_importance']:.4f}")
        print(f"  Arousal importance: {row['arousal_importance']:.4f}")
        print(f"  Dominance importance: {row['dominance_importance']:.4f}")
        print(f"  Most important feature: {['Valence', 'Arousal', 'Dominance'][np.argmax([row['valence_importance'], row['arousal_importance'], row['dominance_importance']])]}")
        print()

def main():
    """
    Main function to evaluate model performance.
    """
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'emotion_classification', 'models')
    data_file = os.path.join(base_dir, 'emotion_classification', 'vad_with_emotions.csv')
    output_dir = os.path.join(base_dir, 'evaluation', 'results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Evaluating model performance...")
    
    # Evaluate all models
    evaluate_all_models(models_dir, data_file, output_dir)
    
    # Analyze feature importance
    analyze_feature_importance(models_dir, output_dir)
    
    print("\nModel evaluation completed.")

if __name__ == "__main__":
    main()

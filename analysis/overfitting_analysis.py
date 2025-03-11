#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis script to evaluate model performance and check for overfitting.
This script analyzes the emotion recognition model to identify overfitting issues.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotion_classification.vad_to_emotion_classifier import VADToEmotionClassifier

def load_data():
    """
    Load the VAD data with emotion labels.
    
    Returns:
        DataFrame with VAD values and emotion labels
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'emotion_classification')
    df = pd.read_csv(os.path.join(data_dir, 'vad_with_emotions.csv'))
    return df

def analyze_data_distribution(df, output_dir):
    """
    Analyze the distribution of VAD values and emotions.
    
    Args:
        df: DataFrame with VAD values and emotion labels
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze VAD value distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    sns.histplot(df['valence_norm'], kde=True)
    plt.title('Valence Distribution')
    plt.xlabel('Valence')
    
    plt.subplot(132)
    sns.histplot(df['activation_norm'], kde=True)
    plt.title('Activation Distribution')
    plt.xlabel('Activation')
    
    plt.subplot(133)
    sns.histplot(df['dominance_norm'], kde=True)
    plt.title('Dominance Distribution')
    plt.xlabel('Dominance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vad_distributions.png'))
    plt.close()
    
    # Analyze emotion distributions for each model
    emotion_models = ['quadrant', 'custom', 'plutchik', 'ekman']
    
    plt.figure(figsize=(15, 10))
    
    for i, model in enumerate(emotion_models):
        plt.subplot(2, 2, i+1)
        sns.countplot(y=df[f'emotion_{model}'], order=df[f'emotion_{model}'].value_counts().index)
        plt.title(f'{model.capitalize()} Emotion Distribution')
        plt.xlabel('Count')
        plt.ylabel('Emotion')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_distributions.png'))
    plt.close()
    
    # Create a report of the data distribution
    with open(os.path.join(output_dir, 'data_distribution_report.txt'), 'w') as f:
        f.write("VAD Value Statistics:\n")
        f.write("====================\n\n")
        
        for dim in ['valence_norm', 'activation_norm', 'dominance_norm']:
            f.write(f"{dim.split('_')[0].capitalize()} statistics:\n")
            f.write(f"  Mean: {df[dim].mean():.4f}\n")
            f.write(f"  Std: {df[dim].std():.4f}\n")
            f.write(f"  Min: {df[dim].min():.4f}\n")
            f.write(f"  25%: {df[dim].quantile(0.25):.4f}\n")
            f.write(f"  Median: {df[dim].median():.4f}\n")
            f.write(f"  75%: {df[dim].quantile(0.75):.4f}\n")
            f.write(f"  Max: {df[dim].max():.4f}\n\n")
        
        f.write("\nEmotion Distributions:\n")
        f.write("====================\n\n")
        
        for model in emotion_models:
            f.write(f"{model.capitalize()} emotion distribution:\n")
            dist = df[f'emotion_{model}'].value_counts()
            for emotion, count in dist.items():
                f.write(f"  {emotion}: {count} ({count/len(df)*100:.2f}%)\n")
            f.write("\n")

def analyze_vad_to_emotion_mapping(df, output_dir):
    """
    Analyze how VAD values are mapped to emotions.
    
    Args:
        df: DataFrame with VAD values and emotion labels
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze the relationship between VAD values and emotions for each model
    emotion_models = ['quadrant', 'custom', 'plutchik', 'ekman']
    
    for model in emotion_models:
        # Create 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        emotions = df[f'emotion_{model}'].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(emotions)))
        
        for i, emotion in enumerate(emotions):
            subset = df[df[f'emotion_{model}'] == emotion]
            ax.scatter(
                subset['valence_norm'], 
                subset['activation_norm'], 
                subset['dominance_norm'],
                c=[colors[i]],
                label=emotion,
                alpha=0.7
            )
        
        ax.set_xlabel('Valence')
        ax.set_ylabel('Activation')
        ax.set_zlabel('Dominance')
        ax.set_title(f'3D VAD Space - {model.capitalize()} Emotions')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'vad_to_emotion_3d_{model}.png'))
        plt.close()
        
        # Create 2D scatter plots for each pair of dimensions
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Valence-Activation
        for i, emotion in enumerate(emotions):
            subset = df[df[f'emotion_{model}'] == emotion]
            axes[0].scatter(
                subset['valence_norm'], 
                subset['activation_norm'],
                c=[colors[i]],
                label=emotion,
                alpha=0.7
            )
        
        axes[0].set_xlabel('Valence')
        axes[0].set_ylabel('Activation')
        axes[0].set_title(f'Valence-Activation Space - {model.capitalize()}')
        axes[0].grid(True)
        
        # Valence-Dominance
        for i, emotion in enumerate(emotions):
            subset = df[df[f'emotion_{model}'] == emotion]
            axes[1].scatter(
                subset['valence_norm'], 
                subset['dominance_norm'],
                c=[colors[i]],
                label=emotion,
                alpha=0.7
            )
        
        axes[1].set_xlabel('Valence')
        axes[1].set_ylabel('Dominance')
        axes[1].set_title(f'Valence-Dominance Space - {model.capitalize()}')
        axes[1].grid(True)
        
        # Activation-Dominance
        for i, emotion in enumerate(emotions):
            subset = df[df[f'emotion_{model}'] == emotion]
            axes[2].scatter(
                subset['activation_norm'], 
                subset['dominance_norm'],
                c=[colors[i]],
                label=emotion,
                alpha=0.7
            )
        
        axes[2].set_xlabel('Activation')
        axes[2].set_ylabel('Dominance')
        axes[2].set_title(f'Activation-Dominance Space - {model.capitalize()}')
        axes[2].grid(True)
        
        # Add legend to the right of the last subplot
        handles, labels = axes[2].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend
        plt.savefig(os.path.join(output_dir, f'vad_to_emotion_2d_{model}.png'))
        plt.close()

def evaluate_model_performance(df, output_dir):
    """
    Evaluate the performance of the VAD-to-emotion classifiers.
    
    Args:
        df: DataFrame with VAD values and emotion labels
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate models for each emotion model
    emotion_models = ['quadrant', 'custom', 'plutchik', 'ekman']
    classifier_types = ['random_forest', 'svm', 'mlp']
    
    results = []
    
    for emotion_model in emotion_models:
        emotion_column = f'emotion_{emotion_model}'
        
        # Prepare features and target
        X = df[['valence_norm', 'activation_norm', 'dominance_norm']]
        y = df[emotion_column]
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for classifier_type in classifier_types:
            # Create classifier
            classifier = VADToEmotionClassifier(
                classifier_type=classifier_type,
                emotion_model=emotion_model
            )
            
            # Load model
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'emotion_classification', 'models',
                f'{emotion_model}_{classifier_type}'
            )
            
            try:
                classifier = VADToEmotionClassifier.load_model(model_dir)
                
                # Evaluate on test set
                y_pred = classifier.classifier.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                # Evaluate on training set
                y_train_pred = classifier.classifier.predict(X_train_scaled)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                
                # Calculate overfitting ratio (train accuracy / test accuracy)
                overfitting_ratio = train_accuracy / test_accuracy if test_accuracy > 0 else float('inf')
                
                # Store results
                results.append({
                    'emotion_model': emotion_model,
                    'classifier_type': classifier_type,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'overfitting_ratio': overfitting_ratio
                })
                
                # Generate confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=np.unique(y), yticklabels=np.unique(y))
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - {classifier_type.upper()} - {emotion_model.upper()}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'confusion_matrix_{emotion_model}_{classifier_type}.png'))
                plt.close()
                
                # Generate classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                report_df.to_csv(os.path.join(output_dir, f'classification_report_{emotion_model}_{classifier_type}.csv'))
                
            except Exception as e:
                print(f"Error evaluating {emotion_model} with {classifier_type}: {e}")
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'model_evaluation_results.csv'), index=False)
    
    # Plot train vs test accuracy
    plt.figure(figsize=(12, 8))
    
    # Group by emotion model and classifier type
    grouped = results_df.groupby(['emotion_model', 'classifier_type'])
    
    # Set up the plot
    bar_width = 0.35
    index = np.arange(len(results_df))
    
    # Create labels for x-axis
    labels = [f"{row['emotion_model']}\n{row['classifier_type']}" for _, row in results_df.iterrows()]
    
    # Plot train and test accuracy
    plt.bar(index - bar_width/2, results_df['train_accuracy'], bar_width, label='Train Accuracy')
    plt.bar(index + bar_width/2, results_df['test_accuracy'], bar_width, label='Test Accuracy')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.xticks(index, labels, rotation=45)
    plt.ylim(0.95, 1.01)  # Adjust y-axis to better show differences
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_vs_test_accuracy.png'))
    plt.close()
    
    # Plot overfitting ratio
    plt.figure(figsize=(12, 8))
    plt.bar(index, results_df['overfitting_ratio'])
    plt.xlabel('Model')
    plt.ylabel('Overfitting Ratio (Train Accuracy / Test Accuracy)')
    plt.title('Overfitting Ratio by Model')
    plt.xticks(index, labels, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overfitting_ratio.png'))
    plt.close()
    
    # Create a report of the model evaluation
    with open(os.path.join(output_dir, 'model_evaluation_report.txt'), 'w') as f:
        f.write("Model Evaluation Report\n")
        f.write("=====================\n\n")
        
        f.write("Summary of Results:\n")
        f.write("-----------------\n\n")
        
        for _, row in results_df.iterrows():
            f.write(f"Emotion model: {row['emotion_model'].upper()}, Classifier: {row['classifier_type'].upper()}\n")
            f.write(f"  Train Accuracy: {row['train_accuracy']:.4f}\n")
            f.write(f"  Test Accuracy: {row['test_accuracy']:.4f}\n")
            f.write(f"  Overfitting Ratio: {row['overfitting_ratio']:.4f}\n\n")
        
        f.write("\nAnalysis of Overfitting:\n")
        f.write("----------------------\n\n")
        
        # Calculate average overfitting ratio
        avg_ratio = results_df['overfitting_ratio'].mean()
        f.write(f"Average Overfitting Ratio: {avg_ratio:.4f}\n\n")
        
        if avg_ratio > 1.01:
            f.write("The models show signs of overfitting, with train accuracy consistently higher than test accuracy.\n")
        else:
            f.write("The models do not show significant signs of overfitting based on accuracy metrics alone.\n")
        
        f.write("\nHowever, the extremely high accuracy (near 100%) across all models suggests other issues:\n\n")
        f.write("1. The task may be too simple or deterministic (e.g., rule-based mapping from VAD to emotions)\n")
        f.write("2. There may be data leakage between training and testing sets\n")
        f.write("3. The evaluation methodology may not be robust (e.g., using the same data for training and evaluation)\n")
        f.write("4. The features (VAD values) may perfectly separate the emotion classes by design\n\n")
        
        f.write("Recommendations:\n")
        f.write("--------------\n\n")
        f.write("1. Implement proper cross-validation with stratified sampling\n")
        f.write("2. Use a separate holdout test set that is never seen during training\n")
        f.write("3. Investigate the relationship between VAD values and emotion labels to understand if the mapping is deterministic\n")
        f.write("4. If using text modality, ensure actual text data is used rather than placeholders\n")
        f.write("5. Consider more challenging evaluation scenarios, such as cross-dataset evaluation\n")

def main():
    """
    Main function to analyze model performance and check for overfitting.
    """
    # Set paths
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Analyzing model performance and checking for overfitting...")
    
    # Load data
    df = load_data()
    
    # Analyze data distribution
    analyze_data_distribution(df, output_dir)
    
    # Analyze VAD to emotion mapping
    analyze_vad_to_emotion_mapping(df, output_dir)
    
    # Evaluate model performance
    evaluate_model_performance(df, output_dir)
    
    print(f"Analysis completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()

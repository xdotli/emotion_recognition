#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modified parameters for emotion recognition model to address overfitting.
This script implements parameter modifications and feature engineering to improve the model.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotion_classification.improved_vad_to_emotion_classifier import ImprovedVADToEmotionClassifier

def add_feature_interactions(X):
    """
    Add feature interactions to the dataset.
    
    Args:
        X: DataFrame with VAD values
        
    Returns:
        DataFrame with original features and interactions
    """
    # Create polynomial features (interactions)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Create new feature names
    feature_names = X.columns.tolist()
    interaction_names = []
    
    # Get the indices of the interaction terms
    for i, name1 in enumerate(feature_names):
        for j, name2 in enumerate(feature_names):
            if j > i:  # Only include interactions once
                interaction_names.append(f"{name1}_{name2}")
    
    # Create new DataFrame with original and interaction features
    X_with_interactions = pd.DataFrame(
        X_poly, 
        columns=feature_names + interaction_names
    )
    
    return X_with_interactions

def add_vad_ratios(X):
    """
    Add VAD ratios as additional features.
    
    Args:
        X: DataFrame with VAD values
        
    Returns:
        DataFrame with original features and VAD ratios
    """
    X_with_ratios = X.copy()
    
    # Add ratios (avoiding division by zero)
    epsilon = 1e-10  # Small value to avoid division by zero
    
    X_with_ratios['valence_arousal_ratio'] = X['valence_norm'] / (X['activation_norm'] + epsilon)
    X_with_ratios['valence_dominance_ratio'] = X['valence_norm'] / (X['dominance_norm'] + epsilon)
    X_with_ratios['arousal_dominance_ratio'] = X['activation_norm'] / (X['dominance_norm'] + epsilon)
    
    return X_with_ratios

def add_vad_differences(X):
    """
    Add VAD differences as additional features.
    
    Args:
        X: DataFrame with VAD values
        
    Returns:
        DataFrame with original features and VAD differences
    """
    X_with_diffs = X.copy()
    
    # Add differences
    X_with_diffs['valence_arousal_diff'] = X['valence_norm'] - X['activation_norm']
    X_with_diffs['valence_dominance_diff'] = X['valence_norm'] - X['dominance_norm']
    X_with_diffs['arousal_dominance_diff'] = X['activation_norm'] - X['dominance_norm']
    
    return X_with_diffs

def select_best_features(X, y, k=6):
    """
    Select the best k features using ANOVA F-value.
    
    Args:
        X: DataFrame with features
        y: Series with target values
        k: Number of features to select
        
    Returns:
        DataFrame with selected features
    """
    # Ensure k is not larger than the number of features
    k = min(k, X.shape[1])
    
    # Select k best features
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()
    
    # Create new DataFrame with selected features
    X_best = pd.DataFrame(X_selected, columns=selected_features)
    
    print(f"Selected {k} best features: {selected_features}")
    
    return X_best

def train_with_modified_parameters(X, y, emotion_model, classifier_type, output_dir):
    """
    Train a model with modified parameters.
    
    Args:
        X: DataFrame with features
        y: Series with target values
        emotion_model: Emotion model name
        classifier_type: Classifier type
        output_dir: Directory to save results
        
    Returns:
        Dictionary with training results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize classifier with modified parameters
    if classifier_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42,
            oob_score=True
        )
        
        # Parameter grid for grid search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
    elif classifier_type == 'svm':
        classifier = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        
        # Parameter grid for grid search
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'degree': [2, 3] # For poly kernel
        }
        
    elif classifier_type == 'mlp':
        classifier = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        
        # Parameter grid for grid search
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'adaptive'],
            'batch_size': [32, 64, 128]
        }
        
    elif classifier_type == 'gradient_boosting':
        classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        
        # Parameter grid for grid search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.7, 0.8, 0.9]
        }
    
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        classifier, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best classifier
    best_classifier = grid_search.best_estimator_
    
    # Evaluate on validation set
    y_val_pred = best_classifier.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    # Evaluate on test set
    y_test_pred = best_classifier.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Print evaluation results
    print(f"Validation accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
    
    print("\nClassification report (validation set):")
    print(classification_report(y_val, y_val_pred))
    
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_test_pred))
    
    # Plot confusion matrices
    plt.figure(figsize=(12, 5))
    
    # Validation set confusion matrix
    plt.subplot(121)
    cm_val = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', 
               xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Validation Set Confusion Matrix\n{classifier_type.upper()} - {emotion_model.upper()}')
    
    # Test set confusion matrix
    plt.subplot(122)
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
               xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Test Set Confusion Matrix\n{classifier_type.upper()} - {emotion_model.upper()}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'modified_confusion_matrix_{emotion_model}_{classifier_type}.png'))
    plt.close()
    
    # Save model
    model_dir = os.path.join(output_dir, f'{emotion_model}_{classifier_type}')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save classifier
    joblib.dump(best_classifier, os.path.join(model_dir, 'classifier.pkl'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    # Save configuration
    config = {
        'classifier_type': classifier_type,
        'emotion_model': emotion_model,
        'best_params': grid_search.best_params_,
        'feature_names': X.columns.tolist()
    }
    
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    # Return training results
    return {
        'emotion_model': emotion_model,
        'classifier_type': classifier_type,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'best_params': grid_search.best_params_
    }

def main():
    """
    Main function to train models with modified parameters.
    """
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    emotion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'emotion_classification')
    output_dir = os.path.join(emotion_dir, 'modified_models')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAD annotations with emotion labels
    df = pd.read_csv(os.path.join(emotion_dir, 'vad_with_emotions.csv'))
    
    print("Training models with modified parameters...")
    print(f"Number of utterances: {len(df)}")
    
    # Define emotion models and classifier types to evaluate
    emotion_models = ['quadrant', 'custom', 'plutchik', 'ekman']
    classifier_types = ['random_forest', 'svm', 'mlp', 'gradient_boosting']
    
    # Store results
    all_results = []
    
    # Train and evaluate models with different feature sets
    for emotion_model in emotion_models:
        emotion_column = f'emotion_{emotion_model}'
        
        # Prepare base features
        X_base = df[['valence_norm', 'activation_norm', 'dominance_norm']]
        y = df[emotion_column]
        
        print(f"\n\n{'='*50}")
        print(f"Training models for {emotion_model.upper()} emotion model")
        print(f"{'='*50}")
        print(f"Emotion distribution:")
        print(y.value_counts())
        print(f"{'='*50}\n")
        
        # 1. Base features
        print(f"\n{'-'*50}")
        print(f"Using base features")
        print(f"{'-'*50}")
        
        for classifier_type in classifier_types:
            print(f"\nTraining {classifier_type.upper()} classifier with base features")
            
            results = train_with_modified_parameters(
                X_base, y, emotion_model, classifier_type, 
                os.path.join(output_dir, 'base_features')
            )
            
            results['feature_set'] = 'base'
            all_results.append(results)
        
        # 2. Features with interactions
        print(f"\n{'-'*50}")
        print(f"Using features with interactions")
        print(f"{'-'*50}")
        
        X_interactions = add_feature_interactions(X_base)
        
        for classifier_type in classifier_types:
            print(f"\nTraining {classifier_type.upper()} classifier with interaction features")
            
            results = train_with_modified_parameters(
                X_interactions, y, emotion_model, classifier_type, 
                os.path.join(output_dir, 'interaction_features')
            )
            
            results['feature_set'] = 'interactions'
            all_results.append(results)
        
        # 3. Features with VAD ratios
        print(f"\n{'-'*50}")
        print(f"Using features with VAD ratios")
        print(f"{'-'*50}")
        
        X_ratios = add_vad_ratios(X_base)
        
        for classifier_type in classifier_types:
            print(f"\nTraining {classifier_type.upper()} classifier with VAD ratio features")
            
            results = train_with_modified_parameters(
                X_ratios, y, emotion_model, classifier_type, 
                os.path.join(output_dir, 'ratio_features')
            )
            
            results['feature_set'] = 'ratios'
            all_results.append(results)
        
        # 4. Features with VAD differences
        print(f"\n{'-'*50}")
        print(f"Using features with VAD differences")
        print(f"{'-'*50}")
        
        X_diffs = add_vad_differences(X_base)
        
        for classifier_type in classifier_types:
            print(f"\nTraining {classifier_type.upper()} classifier with VAD difference features")
            
            results = train_with_modified_parameters(
                X_diffs, y, emotion_model, classifier_type, 
                os.path.join(output_dir, 'difference_features')
            )
            
            results['feature_set'] = 'differences'
            all_results.append(results)
        
        # 5. All features combined
        print(f"\n{'-'*50}")
        print(f"Using all features combined")
        print(f"{'-'*50}")
        
        # Combine all feature sets
        X_all = X_base.copy()
        
        # Add interaction features
        for col in X_interactions.columns:
            if col not in X_all.columns:
                X_all[col] = X_interactions[col]
        
        # Add ratio features
        for col in X_ratios.columns:
            if col not in X_all.columns:
                X_all[col] = X_ratios[col]
        
        # Add difference features
        for col in X_diffs.columns:
            if col not in X_all.columns:
                X_all[col] = X_diffs[col]
        
        for classifier_type in classifier_types:
            print(f"\nTraining {classifier_type.upper()} classifier with all features")
            
            results = train_with_modified_parameters(
                X_all, y, emotion_model, classifier_type, 
                os.path.join(output_dir, 'all_features')
            )
            
            results['feature_set'] = 'all'
            all_results.append(results)
        
        # 6. Best selected features
        print(f"\n{'-'*50}")
        print(f"Using best selected features")
        print(f"{'-'*50}")
        
        X_best = select_best_features(X_all, y, k=6)
        
        for classifier_type in classifier_types:
            print(f"\nTraining {classifier_type.upper()} classifier with best selected features")
            
            results = train_with_modified_parameters(
                X_best, y, emotion_model, classifier_type, 
                os.path.join(output_dir, 'best_features')
            )
            
            results['feature_set'] = 'best'
            all_results.append(results)
    
    # Create DataFrame with all results
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'modified_parameters_results.csv'), index=False)
    print(f"Results saved to {os.path.join(output_dir, 'modified_parameters_results.csv')}")
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # Plot test F1 scores by feature set
    plt.subplot(221)
    sns.barplot(x='feature_set', y='test_f1', hue='classifier_type', data=results_df)
    plt.title('Test F1 Score by Feature Set and Classifier')
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot test F1 scores by emotion model
    plt.subplot(222)
    sns.barplot(x='emotion_model', y='test_f1', hue='classifier_type', data=results_df)
    plt.title('Test F1 Score by Emotion Model and Classifier')
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot validation vs test F1 scores
    plt.subplot(223)
    
    # Prepare data for grouped bar plot
    val_f1s = []
    test_f1s = []
    labels = []
    
    # Get top 10 models by test F1 score
    top_models = results_df.sort_values('test_f1', ascending=False).head(10)
    
    for _, row in top_models.iterrows():
        val_f1s.append(row['val_f1'])
        test_f1s.append(row['test_f1'])
        labels.append(f"{row['emotion_model']}\n{row['classifier_type']}\n{row['feature_set']}")
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, val_f1s, width, label='Validation F1')
    plt.bar(x + width/2, test_f1s, width, label='Test F1')
    
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title('Top 10 Models: Validation vs Test F1 Score')
    plt.xticks(x, labels, rotation=45)
    plt.ylim(0.5, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot test accuracy vs F1 score
    plt.subplot(224)
    plt.scatter(results_df['test_accuracy'], results_df['test_f1'], 
               c=results_df['emotion_model'].astype('category').cat.codes, 
               alpha=0.7)
    
    plt.xlabel('Test Accuracy')
    plt.ylabel('Test F1 Score')
    plt.title('Test Accuracy vs F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'modified_parameters_comparison.png'))
    plt.close()
    
    print(f"Comparison plots saved to {os.path.join(output_dir, 'modified_parameters_comparison.png')}")
    
    # Print summary of best models
    print("\n\nSummary of best models by feature set:")
    print(f"{'='*70}")
    
    for feature_set in results_df['feature_set'].unique():
        subset = results_df[results_df['feature_set'] == feature_set]
        best_model = subset.loc[subset['test_f1'].idxmax()]
        
        print(f"Feature set: {feature_set}")
        print(f"  Best model: {best_model['emotion_model'].upper()} with {best_model['classifier_type'].upper()}")
        print(f"  Test F1: {best_model['test_f1']:.4f}, Test Accuracy: {best_model['test_accuracy']:.4f}")
        print(f"  Best parameters: {best_model['best_params']}")
        print()
    
    # Find overall best model
    best_model = results_df.loc[results_df['test_f1'].idxmax()]
    print(f"\nOverall best model:")
    print(f"  Emotion model: {best_model['emotion_model'].upper()}")
    print(f"  Classifier: {best_model['classifier_type'].upper()}")
    print(f"  Feature set: {best_model['feature_set']}")
    print(f"  Test F1: {best_model['test_f1']:.4f}, Test Accuracy: {best_model['test_accuracy']:.4f}")
    print(f"  Best parameters: {best_model['best_params']}")

if __name__ == "__main__":
    main()

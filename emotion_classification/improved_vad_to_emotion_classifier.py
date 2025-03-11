#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved VAD to emotion classifier implementation with proper cross-validation.
This script implements a classifier that maps VAD (Valence-Arousal-Dominance) values to emotion categories
with improved validation methodology to address overfitting issues.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib
import json

class ImprovedVADToEmotionClassifier:
    def __init__(self, classifier_type='random_forest', emotion_model='ekman'):
        """
        Initialize the improved VAD to emotion classifier.
        
        Args:
            classifier_type: Type of classifier to use ('random_forest', 'svm', or 'mlp')
            emotion_model: Emotion model to use ('quadrant', 'custom', 'plutchik', or 'ekman')
        """
        self.classifier_type = classifier_type
        self.emotion_model = emotion_model
        self.classifier = None
        self.scaler = StandardScaler()
        
        # Set up classifier with regularization to prevent overfitting
        if classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                random_state=42,
                # Add regularization parameters
                max_features='sqrt',  # Limit features considered at each split
                min_samples_leaf=2,   # Require at least 2 samples in leaf nodes
                oob_score=True        # Use out-of-bag samples to estimate performance
            )
        elif classifier_type == 'svm':
            self.classifier = SVC(
                probability=True, 
                random_state=42,
                # Add regularization parameters
                C=1.0,                # Smaller C means stronger regularization
                class_weight='balanced' # Handle class imbalance
            )
        elif classifier_type == 'mlp':
            self.classifier = MLPClassifier(
                random_state=42, 
                max_iter=1000,
                # Add regularization parameters
                alpha=0.01,           # L2 regularization parameter
                early_stopping=True,  # Stop training when validation score doesn't improve
                validation_fraction=0.1 # Fraction of training data for validation
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    def train(self, X, y, param_grid=None):
        """
        Train the classifier on VAD values and emotion labels with proper cross-validation.
        
        Args:
            X: DataFrame with VAD values
            y: Series with emotion labels
            param_grid: Dictionary with hyperparameters for grid search
            
        Returns:
            Dictionary with training results including cross-validation scores
        """
        # Create a proper train/validation/test split
        # First split off the test set (20% of data)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Then split the remaining data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # Scale features using only training data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Default parameter grids if none provided
        if param_grid is None:
            if self.classifier_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'max_features': ['sqrt', 'log2'],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.classifier_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear'],
                    'class_weight': ['balanced', None]
                }
            elif self.classifier_type == 'mlp':
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'adaptive'],
                    'early_stopping': [True]
                }
        
        # Set up cross-validation with stratified k-fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            self.classifier, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best classifier
        self.classifier = grid_search.best_estimator_
        
        # Evaluate on validation set
        y_val_pred = self.classifier.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        # Evaluate on test set
        y_test_pred = self.classifier.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Perform k-fold cross-validation on the entire dataset
        X_full_scaled = self.scaler.transform(X)
        cv_scores = cross_val_score(
            self.classifier, X_full_scaled, y, cv=cv, scoring='f1_weighted'
        )
        
        # Print evaluation results
        print(f"Validation accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
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
        plt.title(f'Validation Set Confusion Matrix\n{self.classifier_type.upper()} - {self.emotion_model.upper()}')
        
        # Test set confusion matrix
        plt.subplot(122)
        cm_test = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Test Set Confusion Matrix\n{self.classifier_type.upper()} - {self.emotion_model.upper()}')
        
        plt.tight_layout()
        plt.savefig(f'improved_confusion_matrix_{self.classifier_type}_{self.emotion_model}.png')
        plt.close()
        
        # Return training results
        return {
            'best_params': grid_search.best_params_,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def predict(self, X):
        """
        Predict emotion categories from VAD values.
        
        Args:
            X: DataFrame with VAD values
            
        Returns:
            DataFrame with predicted emotions and probabilities
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred = self.classifier.predict(X_scaled)
        
        # Get probabilities
        y_proba = self.classifier.predict_proba(X_scaled)
        
        # Create DataFrame with predictions
        df_pred = pd.DataFrame({'predicted_emotion': y_pred})
        
        # Add probabilities for each class
        for i, emotion in enumerate(self.classifier.classes_):
            df_pred[f'prob_{emotion}'] = y_proba[:, i]
        
        return df_pred
    
    def save_model(self, output_dir):
        """
        Save the model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save classifier
        joblib.dump(self.classifier, os.path.join(output_dir, 'classifier.pkl'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        
        # Save configuration
        config = {
            'classifier_type': self.classifier_type,
            'emotion_model': self.emotion_model
        }
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
    
    @classmethod
    def load_model(cls, model_dir):
        """
        Load a saved model from disk.
        
        Args:
            model_dir: Directory where the model is saved
            
        Returns:
            Loaded ImprovedVADToEmotionClassifier
        """
        # Load configuration
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Create classifier
        classifier = cls(
            classifier_type=config['classifier_type'],
            emotion_model=config['emotion_model']
        )
        
        # Load classifier
        classifier.classifier = joblib.load(os.path.join(model_dir, 'classifier.pkl'))
        
        # Load scaler
        classifier.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        return classifier

def main():
    """
    Main function to train and evaluate the improved VAD to emotion classifier.
    """
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    emotion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'emotion_classification')
    output_dir = os.path.join(emotion_dir, 'improved_models')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAD annotations with emotion labels
    df = pd.read_csv(os.path.join(emotion_dir, 'vad_with_emotions.csv'))
    
    print("Implementing improved VAD to emotion classifier with proper cross-validation...")
    print(f"Number of utterances: {len(df)}")
    
    # Define emotion models and classifier types to evaluate
    emotion_models = ['quadrant', 'custom', 'plutchik', 'ekman']
    classifier_types = ['random_forest', 'svm', 'mlp']
    
    # Store results
    results = []
    
    # Train and evaluate classifiers
    for emotion_model in emotion_models:
        emotion_column = f'emotion_{emotion_model}'
        
        # Prepare features and target
        X = df[['valence_norm', 'activation_norm', 'dominance_norm']]
        y = df[emotion_column]
        
        print(f"\n\n{'='*50}")
        print(f"Training improved classifiers for {emotion_model.upper()} emotion model")
        print(f"{'='*50}")
        print(f"Emotion distribution:")
        print(y.value_counts())
        print(f"{'='*50}\n")
        
        for classifier_type in classifier_types:
            print(f"\n{'-'*50}")
            print(f"Training improved {classifier_type.upper()} classifier")
            print(f"{'-'*50}")
            
            # Create and train classifier
            classifier = ImprovedVADToEmotionClassifier(
                classifier_type=classifier_type,
                emotion_model=emotion_model
            )
            
            # Train classifier with proper cross-validation
            training_results = classifier.train(X, y)
            
            # Save model
            model_dir = os.path.join(output_dir, f'{emotion_model}_{classifier_type}')
            classifier.save_model(model_dir)
            print(f"Model saved to {model_dir}")
            
            # Store results
            results.append({
                'emotion_model': emotion_model,
                'classifier_type': classifier_type,
                'val_accuracy': training_results['val_accuracy'],
                'val_f1': training_results['val_f1'],
                'test_accuracy': training_results['test_accuracy'],
                'test_f1': training_results['test_f1'],
                'cv_mean': training_results['cv_mean'],
                'cv_std': training_results['cv_std'],
                'best_params': training_results['best_params']
            })
    
    # Print summary of results
    print("\n\nSummary of improved classifier results:")
    print(f"{'='*70}")
    for result in sorted(results, key=lambda x: x['test_f1'], reverse=True):
        print(f"Emotion model: {result['emotion_model'].upper()}, "
              f"Classifier: {result['classifier_type'].upper()}")
        print(f"  Validation - Accuracy: {result['val_accuracy']:.4f}, F1: {result['val_f1']:.4f}")
        print(f"  Test - Accuracy: {result['test_accuracy']:.4f}, F1: {result['test_f1']:.4f}")
        print(f"  Cross-validation - F1: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        print(f"  Best parameters: {result['best_params']}")
        print()
    
    # Find best model
    best_result = max(results, key=lambda x: x['test_f1'])
    print(f"\nBest model: {best_result['emotion_model'].upper()} with "
          f"{best_result['classifier_type'].upper()} classifier "
          f"(Test F1: {best_result['test_f1']:.4f})")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'improved_classifier_results.csv'), index=False)
    print(f"Results saved to {os.path.join(output_dir, 'improved_classifier_results.csv')}")
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # Plot test F1 scores
    plt.subplot(221)
    sns.barplot(x='emotion_model', y='test_f1', hue='classifier_type', data=results_df)
    plt.title('Test F1 Score by Model and Classifier')
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot test accuracy
    plt.subplot(222)
    sns.barplot(x='emotion_model', y='test_accuracy', hue='classifier_type', data=results_df)
    plt.title('Test Accuracy by Model and Classifier')
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot cross-validation F1 scores
    plt.subplot(223)
    sns.barplot(x='emotion_model', y='cv_mean', hue='classifier_type', data=results_df)
    plt.title('Cross-Validation F1 Score by Model and Classifier')
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot validation vs test accuracy
    plt.subplot(224)
    
    # Prepare data for grouped bar plot
    models = []
    val_accuracies = []
    test_accuracies = []
    labels = []
    
    for _, row in results_df.iterrows():
        models.append(f"{row['emotion_model']}_{row['classifier_type']}")
        val_accuracies.append(row['val_accuracy'])
        test_accuracies.append(row['test_accuracy'])
        labels.append(f"{row['emotion_model']}\n{row['classifier_type']}")
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, val_accuracies, width, label='Validation')
    plt.bar(x + width/2, test_accuracies, width, label='Test')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Validation vs Test Accuracy')
    plt.xticks(x, labels, rotation=45)
    plt.ylim(0.5, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improved_classifier_comparison.png'))
    plt.close()
    
    print(f"Comparison plots saved to {os.path.join(output_dir, 'improved_classifier_comparison.png')}")

if __name__ == "__main__":
    main()

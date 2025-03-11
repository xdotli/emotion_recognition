#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VAD to emotion classifier implementation.
This script implements a classifier that maps VAD (Valence-Arousal-Dominance) values to emotion categories.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib
import json

class VADToEmotionClassifier:
    def __init__(self, classifier_type='random_forest', emotion_model='ekman'):
        """
        Initialize the VAD to emotion classifier.
        
        Args:
            classifier_type: Type of classifier to use ('random_forest', 'svm', or 'mlp')
            emotion_model: Emotion model to use ('quadrant', 'custom', 'plutchik', or 'ekman')
        """
        self.classifier_type = classifier_type
        self.emotion_model = emotion_model
        self.classifier = None
        self.scaler = StandardScaler()
        
        # Set up classifier
        if classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(random_state=42)
        elif classifier_type == 'svm':
            self.classifier = SVC(probability=True, random_state=42)
        elif classifier_type == 'mlp':
            self.classifier = MLPClassifier(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    def train(self, X, y, param_grid=None):
        """
        Train the classifier on VAD values and emotion labels.
        
        Args:
            X: DataFrame with VAD values
            y: Series with emotion labels
            param_grid: Dictionary with hyperparameters for grid search
            
        Returns:
            Best parameters from grid search
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Default parameter grids if none provided
        if param_grid is None:
            if self.classifier_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.classifier_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                }
            elif self.classifier_type == 'mlp':
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
        
        # Grid search
        grid_search = GridSearchCV(
            self.classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best classifier
        self.classifier = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {self.classifier_type.upper()} - {self.emotion_model.upper()}')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{self.classifier_type}_{self.emotion_model}.png')
        plt.close()
        
        return grid_search.best_params_
    
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
            Loaded VADToEmotionClassifier
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
    Main function to train and evaluate the VAD to emotion classifier.
    """
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    emotion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'emotion_classification')
    output_dir = os.path.join(emotion_dir, 'models')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAD annotations with emotion labels
    df = pd.read_csv(os.path.join(emotion_dir, 'vad_with_emotions.csv'))
    
    print("Implementing VAD to emotion classifier...")
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
        print(f"Training classifiers for {emotion_model.upper()} emotion model")
        print(f"{'='*50}")
        print(f"Emotion distribution:")
        print(y.value_counts())
        print(f"{'='*50}\n")
        
        for classifier_type in classifier_types:
            print(f"\n{'-'*50}")
            print(f"Training {classifier_type.upper()} classifier")
            print(f"{'-'*50}")
            
            # Create and train classifier
            classifier = VADToEmotionClassifier(
                classifier_type=classifier_type,
                emotion_model=emotion_model
            )
            
            # Train classifier
            best_params = classifier.train(X, y)
            
            # Save model
            model_dir = os.path.join(output_dir, f'{emotion_model}_{classifier_type}')
            classifier.save_model(model_dir)
            print(f"Model saved to {model_dir}")
            
            # Make predictions on the entire dataset
            predictions = classifier.predict(X)
            
            # Calculate accuracy
            accuracy = accuracy_score(y, predictions['predicted_emotion'])
            
            # Store results
            results.append({
                'emotion_model': emotion_model,
                'classifier_type': classifier_type,
                'accuracy': accuracy,
                'best_params': best_params
            })
    
    # Print summary of results
    print("\n\nSummary of results:")
    print(f"{'='*50}")
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"Emotion model: {result['emotion_model'].upper()}, "
              f"Classifier: {result['classifier_type'].upper()}, "
              f"Accuracy: {result['accuracy']:.4f}")
    
    # Find best model
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest model: {best_result['emotion_model'].upper()} with "
          f"{best_result['classifier_type'].upper()} classifier "
          f"(Accuracy: {best_result['accuracy']:.4f})")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'classifier_results.csv'), index=False)
    print(f"Results saved to {os.path.join(output_dir, 'classifier_results.csv')}")

if __name__ == "__main__":
    main()

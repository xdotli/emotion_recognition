#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text to VAD conversion model for emotion recognition.
This script implements a model that converts text to VAD (Valence-Arousal-Dominance) values.
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class TextToVADModel:
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initialize the Text to VAD conversion model.
        
        Args:
            model_name: Name of the pre-trained transformer model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Linear layers for VAD prediction
        self.valence_layer = torch.nn.Linear(768, 1).to(self.device)
        self.arousal_layer = torch.nn.Linear(768, 1).to(self.device)
        self.dominance_layer = torch.nn.Linear(768, 1).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': 1e-5},
            {'params': self.valence_layer.parameters(), 'lr': 1e-4},
            {'params': self.arousal_layer.parameters(), 'lr': 1e-4},
            {'params': self.dominance_layer.parameters(), 'lr': 1e-4}
        ])
        
        # Loss function
        self.criterion = torch.nn.MSELoss()
    
    def get_text_embeddings(self, texts):
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of embeddings
        """
        # Tokenize texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding
        
        return embeddings
    
    def predict_vad(self, texts):
        """
        Predict VAD values for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with predicted VAD values
        """
        # Get embeddings
        embeddings = self.get_text_embeddings(texts)
        
        # Predict VAD values
        valence = self.valence_layer(embeddings).squeeze().cpu().numpy()
        arousal = self.arousal_layer(embeddings).squeeze().cpu().numpy()
        dominance = self.dominance_layer(embeddings).squeeze().cpu().numpy()
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'valence_pred': valence,
            'arousal_pred': arousal,
            'dominance_pred': dominance
        })
        
        return df
    
    def train(self, texts, vad_values, epochs=10, batch_size=16):
        """
        Train the model on texts and their VAD values.
        
        Args:
            texts: List of text strings
            vad_values: DataFrame with columns 'valence_norm', 'activation_norm', 'dominance_norm'
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training history
        """
        # Convert to numpy arrays
        valence = vad_values['valence_norm'].values
        arousal = vad_values['activation_norm'].values
        dominance = vad_values['dominance_norm'].values
        
        # Training history
        history = {
            'val_loss': [],
            'train_loss': [],
            'val_valence_mse': [],
            'val_arousal_mse': [],
            'val_dominance_mse': []
        }
        
        # Split data into train and validation sets
        train_texts, val_texts, train_valence, val_valence, train_arousal, val_arousal, train_dominance, val_dominance = train_test_split(
            texts, valence, arousal, dominance, test_size=0.2, random_state=42
        )
        
        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            self.model.train()
            train_loss = 0
            
            # Process in batches
            for i in range(0, len(train_texts), batch_size):
                batch_texts = train_texts[i:i+batch_size]
                batch_valence = torch.tensor(train_valence[i:i+batch_size], dtype=torch.float32).to(self.device)
                batch_arousal = torch.tensor(train_arousal[i:i+batch_size], dtype=torch.float32).to(self.device)
                batch_dominance = torch.tensor(train_dominance[i:i+batch_size], dtype=torch.float32).to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Get embeddings
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                # Predict VAD values
                pred_valence = self.valence_layer(embeddings).squeeze()
                pred_arousal = self.arousal_layer(embeddings).squeeze()
                pred_dominance = self.dominance_layer(embeddings).squeeze()
                
                # Calculate loss
                loss_valence = self.criterion(pred_valence, batch_valence)
                loss_arousal = self.criterion(pred_arousal, batch_arousal)
                loss_dominance = self.criterion(pred_dominance, batch_dominance)
                loss = loss_valence + loss_arousal + loss_dominance
                
                # Backpropagation
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= (len(train_texts) / batch_size)
            history['train_loss'].append(train_loss)
            
            # Validate
            self.model.eval()
            val_loss = 0
            all_val_valence = []
            all_val_arousal = []
            all_val_dominance = []
            all_pred_valence = []
            all_pred_arousal = []
            all_pred_dominance = []
            
            with torch.no_grad():
                for i in range(0, len(val_texts), batch_size):
                    batch_texts = val_texts[i:i+batch_size]
                    batch_valence = torch.tensor(val_valence[i:i+batch_size], dtype=torch.float32).to(self.device)
                    batch_arousal = torch.tensor(val_arousal[i:i+batch_size], dtype=torch.float32).to(self.device)
                    batch_dominance = torch.tensor(val_dominance[i:i+batch_size], dtype=torch.float32).to(self.device)
                    
                    # Get embeddings
                    inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    
                    # Predict VAD values
                    pred_valence = self.valence_layer(embeddings).squeeze()
                    pred_arousal = self.arousal_layer(embeddings).squeeze()
                    pred_dominance = self.dominance_layer(embeddings).squeeze()
                    
                    # Calculate loss
                    loss_valence = self.criterion(pred_valence, batch_valence)
                    loss_arousal = self.criterion(pred_arousal, batch_arousal)
                    loss_dominance = self.criterion(pred_dominance, batch_dominance)
                    loss = loss_valence + loss_arousal + loss_dominance
                    
                    val_loss += loss.item()
                    
                    # Store predictions and true values
                    all_val_valence.extend(batch_valence.cpu().numpy())
                    all_val_arousal.extend(batch_arousal.cpu().numpy())
                    all_val_dominance.extend(batch_dominance.cpu().numpy())
                    all_pred_valence.extend(pred_valence.cpu().numpy())
                    all_pred_arousal.extend(pred_arousal.cpu().numpy())
                    all_pred_dominance.extend(pred_dominance.cpu().numpy())
            
            val_loss /= (len(val_texts) / batch_size)
            history['val_loss'].append(val_loss)
            
            # Calculate MSE for each dimension
            valence_mse = mean_squared_error(all_val_valence, all_pred_valence)
            arousal_mse = mean_squared_error(all_val_arousal, all_pred_arousal)
            dominance_mse = mean_squared_error(all_val_dominance, all_pred_dominance)
            
            history['val_valence_mse'].append(valence_mse)
            history['val_arousal_mse'].append(arousal_mse)
            history['val_dominance_mse'].append(dominance_mse)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val MSE - Valence: {valence_mse:.4f}, Arousal: {arousal_mse:.4f}, Dominance: {dominance_mse:.4f}")
        
        return history
    
    def save_model(self, output_dir):
        """
        Save the model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model configuration
        config = {
            'model_name': self.model_name
        }
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pt'))
        torch.save(self.valence_layer.state_dict(), os.path.join(output_dir, 'valence_layer.pt'))
        torch.save(self.arousal_layer.state_dict(), os.path.join(output_dir, 'arousal_layer.pt'))
        torch.save(self.dominance_layer.state_dict(), os.path.join(output_dir, 'dominance_layer.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
    
    @classmethod
    def load_model(cls, model_dir):
        """
        Load a saved model from disk.
        
        Args:
            model_dir: Directory where the model is saved
            
        Returns:
            Loaded TextToVADModel
        """
        # Load configuration
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Create model
        model = cls(model_name=config['model_name'])
        
        # Load weights
        model.model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
        model.valence_layer.load_state_dict(torch.load(os.path.join(model_dir, 'valence_layer.pt')))
        model.arousal_layer.load_state_dict(torch.load(os.path.join(model_dir, 'arousal_layer.pt')))
        model.dominance_layer.load_state_dict(torch.load(os.path.join(model_dir, 'dominance_layer.pt')))
        
        return model

def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot MSE
    plt.subplot(1, 2, 2)
    plt.plot(history['val_valence_mse'], label='Valence MSE')
    plt.plot(history['val_arousal_mse'], label='Arousal MSE')
    plt.plot(history['val_dominance_mse'], label='Dominance MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Validation MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    """
    Main function to train and evaluate the Text to VAD model.
    """
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAD annotations
    vad_df = pd.read_csv(os.path.join(processed_dir, 'vad_annotations.csv'))
    
    # Since we don't have the actual text transcriptions, we'll use a pre-trained model approach
    # We'll use the utterance IDs as placeholders for now
    # In a real scenario, you would replace this with actual text data
    texts = vad_df['utterance_id'].tolist()
    
    print("Creating a text-to-VAD model using pre-trained transformers...")
    print(f"Number of utterances: {len(texts)}")
    
    # Create and train model
    model = TextToVADModel()
    
    # In a real scenario with actual text data, you would train the model like this:
    # history = model.train(texts, vad_df, epochs=5, batch_size=16)
    # plot_training_history(history)
    
    # For now, we'll just save the untrained model
    model.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    
    # Example of how to use the model for prediction
    # (using utterance IDs as placeholders for actual text)
    sample_texts = texts[:5]
    predictions = model.predict_vad(sample_texts)
    
    print("\nSample predictions (using utterance IDs as placeholders):")
    print(predictions)
    
    print("\nNote: In a real scenario, you would replace the utterance IDs with actual text data.")
    print("The model architecture is set up to handle text input, but training requires actual text transcriptions.")

if __name__ == "__main__":
    main()

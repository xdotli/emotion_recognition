#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration file for the emotion recognition project.
This module contains configuration parameters used across different components of the project.
"""

# Paths
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# VAD normalization
VAD_MIN = 1
VAD_MAX = 5
VAD_RANGE = VAD_MAX - VAD_MIN

# Emotion models
EMOTION_MODELS = ["quadrant", "custom", "plutchik", "ekman"]

# Classifier types
CLASSIFIER_TYPES = ["random_forest", "svm", "mlp"]

# Quadrant method thresholds
QUADRANT_THRESHOLD = 0.5  # Midpoint for normalized values

# Custom method thresholds
CUSTOM_HIGH_VALENCE = 0.7
CUSTOM_LOW_VALENCE = 0.3
CUSTOM_HIGH_AROUSAL = 0.7
CUSTOM_LOW_AROUSAL = 0.3
CUSTOM_HIGH_DOMINANCE = 0.7
CUSTOM_LOW_DOMINANCE = 0.3
CUSTOM_MID_VALENCE = 0.5
CUSTOM_MID_AROUSAL_LOW = 0.4
CUSTOM_MID_AROUSAL_HIGH = 0.6

# Plutchik's wheel thresholds
PLUTCHIK_HIGH_VALENCE = 0.7
PLUTCHIK_LOW_VALENCE = 0.4
PLUTCHIK_HIGH_AROUSAL = 0.7
PLUTCHIK_LOW_AROUSAL = 0.4
PLUTCHIK_HIGH_DOMINANCE = 0.7
PLUTCHIK_LOW_DOMINANCE = 0.4

# Ekman's basic emotions thresholds
EKMAN_HIGH_VALENCE = 0.7
EKMAN_LOW_VALENCE = 0.3
EKMAN_HIGH_AROUSAL = 0.7
EKMAN_LOW_AROUSAL = 0.4
EKMAN_HIGH_DOMINANCE = 0.6
EKMAN_LOW_DOMINANCE = 0.4
EKMAN_MID_VALENCE = 0.5
EKMAN_MID_AROUSAL = 0.5

# Model training parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Random Forest parameters
RF_N_ESTIMATORS = [50, 100, 200]
RF_MAX_DEPTH = [None, 10, 20]
RF_MIN_SAMPLES_SPLIT = [2, 5, 10]

# SVM parameters
SVM_C = [0.1, 1, 10]
SVM_GAMMA = ['scale', 'auto']
SVM_KERNEL = ['rbf', 'linear']

# MLP parameters
MLP_HIDDEN_LAYER_SIZES = [(50,), (100,), (50, 50)]
MLP_ALPHA = [0.0001, 0.001, 0.01]
MLP_LEARNING_RATE = ['constant', 'adaptive']
MLP_MAX_ITER = 1000

# Visualization parameters
VISUALIZATION_DPI = 300
VISUALIZATION_FIGSIZE_LARGE = (15, 10)
VISUALIZATION_FIGSIZE_MEDIUM = (12, 8)
VISUALIZATION_FIGSIZE_SMALL = (10, 6)
VISUALIZATION_CMAP = 'tab10'

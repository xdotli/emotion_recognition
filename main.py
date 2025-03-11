#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for the emotion recognition project.
This script provides a unified interface to run the entire emotion recognition pipeline.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from utils import create_directory, get_project_root

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Emotion Recognition using VAD Approach')
    parser.add_argument('--preprocess', action='store_true', help='Run data preprocessing')
    parser.add_argument('--text-to-vad', action='store_true', help='Run text to VAD conversion')
    parser.add_argument('--vad-research', action='store_true', help='Run VAD to emotion mapping research')
    parser.add_argument('--vad-classifier', action='store_true', help='Run VAD to emotion classifier')
    parser.add_argument('--evaluate', action='store_true', help='Run model evaluation')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline')
    
    return parser.parse_args()

def run_preprocessing():
    """
    Run data preprocessing.
    """
    print("\n=== Running Data Preprocessing ===")
    from data.preprocess import main as preprocess_main
    preprocess_main()

def run_text_to_vad():
    """
    Run text to VAD conversion.
    """
    print("\n=== Running Text to VAD Conversion ===")
    from vad_conversion.text_to_vad import main as text_to_vad_main
    text_to_vad_main()

def run_vad_research():
    """
    Run VAD to emotion mapping research.
    """
    print("\n=== Running VAD to Emotion Mapping Research ===")
    from emotion_classification.vad_to_emotion_research import main as vad_research_main
    vad_research_main()

def run_vad_classifier():
    """
    Run VAD to emotion classifier.
    """
    print("\n=== Running VAD to Emotion Classifier ===")
    from emotion_classification.vad_to_emotion_classifier import main as vad_classifier_main
    vad_classifier_main()

def run_evaluation():
    """
    Run model evaluation.
    """
    print("\n=== Running Model Evaluation ===")
    from evaluation.model_evaluation import main as evaluation_main
    evaluation_main()

def run_visualization():
    """
    Create visualizations.
    """
    print("\n=== Creating Visualizations ===")
    from visualization.create_visualizations import main as visualization_main
    visualization_main()

def main():
    """
    Main function to run the emotion recognition pipeline.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Add project root to path
    project_root = get_project_root()
    sys.path.append(project_root)
    
    # Run the entire pipeline if --all is specified
    if args.all:
        args.preprocess = True
        args.text_to_vad = True
        args.vad_research = True
        args.vad_classifier = True
        args.evaluate = True
        args.visualize = True
    
    # Run selected components
    if args.preprocess:
        run_preprocessing()
    
    if args.text_to_vad:
        run_text_to_vad()
    
    if args.vad_research:
        run_vad_research()
    
    if args.vad_classifier:
        run_vad_classifier()
    
    if args.evaluate:
        run_evaluation()
    
    if args.visualize:
        run_visualization()
    
    print("\n=== Emotion Recognition Pipeline Completed ===")

if __name__ == "__main__":
    main()

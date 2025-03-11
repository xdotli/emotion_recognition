# Emotion Recognition using IEMOCAP Dataset

This repository contains code for emotion recognition using the IEMOCAP (Interactive Emotional Dyadic Motion Capture) dataset. The approach uses both audio and text modalities in a two-step process:

1. Convert each modality (audio and text) to valence-arousal-dominance (VAD) tuples
2. Categorize emotions based on the VAD tuples

## Project Structure

- `data/`: Scripts for downloading and preprocessing the IEMOCAP dataset
- `models/`: Implementation of audio and text processing models
- `vad_conversion/`: Code for converting modalities to VAD tuples
- `emotion_classification/`: Code for mapping VAD tuples to emotion categories
- `fusion/`: Implementation of multimodal fusion approaches
- `evaluation/`: Scripts for evaluating model performance
- `visualization/`: Code for visualizing results

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Librosa
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Setup

Instructions for setting up the environment and downloading the dataset will be provided.

## Usage

Details on how to train and evaluate the models will be added.

## References

References to related work on two-step emotion recognition approaches will be included.

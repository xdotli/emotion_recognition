# Emotion Recognition using VAD Approach

This repository contains an implementation of a two-step emotion recognition system using the IEMOCAP dataset. The system converts text to Valence-Arousal-Dominance (VAD) tuples and then categorizes emotions based on these tuples.

## Overview

Emotion recognition is a crucial component in human-computer interaction, affective computing, and sentiment analysis. This project implements a two-step approach:

1. **Modality to VAD Conversion**: Converting text to VAD (Valence-Arousal-Dominance) tuples
2. **VAD to Emotion Classification**: Categorizing emotions based on VAD tuples

The implementation focuses on the text modality due to limitations in accessing the complete IEMOCAP dataset.

## Repository Structure

```
emotion_recognition/
├── data/
│   ├── raw/                  # Raw IEMOCAP dataset files
│   ├── processed/            # Processed data files
│   └── preprocess.py         # Data preprocessing script
├── vad_conversion/
│   ├── models/               # Saved text-to-VAD models
│   └── text_to_vad.py        # Text to VAD conversion implementation
├── emotion_classification/
│   ├── models/               # Saved VAD-to-emotion classifiers
│   ├── vad_to_emotion_research.py  # VAD to emotion mapping research
│   └── vad_to_emotion_classifier.py # VAD to emotion classifier implementation
├── evaluation/
│   ├── results/              # Evaluation results
│   └── model_evaluation.py   # Model evaluation script
├── visualization/
│   ├── results/              # Visualization results
│   └── create_visualizations.py # Visualization script
├── docs/
│   └── methodology_and_findings.md # Detailed methodology and findings
├── README.md                 # This file
└── todo.md                   # Project progress tracking
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kookiemaster/emotion_recognition_iemocap.git
cd emotion_recognition_iemocap
```

2. Install the required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch transformers joblib
```

## Usage

### Data Preprocessing

To preprocess the IEMOCAP dataset and extract VAD annotations:

```bash
python data/preprocess.py
```

This script:
- Extracts VAD values from attribute files
- Normalizes the values to a 0-1 range
- Maps VAD values to emotion categories using different approaches
- Saves the processed data to CSV files

### Text to VAD Conversion

To use the text-to-VAD conversion model:

```bash
python vad_conversion/text_to_vad.py
```

This script:
- Implements a text-to-VAD conversion model using pre-trained transformers
- Provides functionality for training, evaluation, and saving/loading models
- Note: Requires text data for actual training

### VAD to Emotion Classification

To research VAD-to-emotion mapping approaches:

```bash
python emotion_classification/vad_to_emotion_research.py
```

To train and evaluate VAD-to-emotion classifiers:

```bash
python emotion_classification/vad_to_emotion_classifier.py
```

These scripts:
- Implement four different VAD-to-emotion mapping approaches
- Train multiple classifier types (Random Forest, SVM, MLP)
- Evaluate classifier performance
- Save trained models

### Model Evaluation

To evaluate model performance:

```bash
python evaluation/model_evaluation.py
```

This script:
- Performs cross-validation
- Analyzes feature importance
- Generates performance metrics
- Creates evaluation visualizations

### Visualization

To create visualizations of the results:

```bash
python visualization/create_visualizations.py
```

This script generates:
- Emotion distribution plots
- 3D VAD scatter plots
- VAD pairplots
- Correlation heatmaps
- Dimension histograms
- Boxplots by emotion
- t-SNE and PCA visualizations

## Emotion Models

The project implements four different approaches for mapping VAD values to emotion categories:

1. **Quadrant Method**: Divides the Valence-Arousal space into four quadrants
   - High Valence, High Arousal → Happy
   - High Valence, Low Arousal → Calm
   - Low Valence, High Arousal → Angry
   - Low Valence, Low Arousal → Sad

2. **Custom Method**: A more nuanced approach considering all three dimensions
   - Seven emotions: Happy, Excited, Content, Neutral, Sad, Afraid, Angry

3. **Plutchik's Wheel**: Based on psychological theory of eight primary emotions
   - Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation

4. **Ekman's Basic Emotions**: Six universal emotions recognized across cultures
   - Happiness, Surprise, Fear, Sadness, Disgust, Anger

## Results

The evaluation shows near-perfect accuracy across all models:
- The Quadrant model with Random Forest classifier achieved 100% accuracy
- Feature importance analysis revealed Valence and Arousal as the most important features
- Detailed results are available in the [methodology and findings document](docs/methodology_and_findings.md)

## Limitations and Future Work

1. **Limited Dataset**: The implementation is limited to VAD annotations without complete text transcriptions or audio files
2. **Text Transcriptions**: Future work could include obtaining complete transcriptions and training the text-to-VAD model
3. **Audio Modality**: Incorporating audio features would provide a more robust system
4. **Multimodal Fusion**: Combining text and audio modalities could improve performance
5. **Real-world Testing**: Testing on real-world data would provide insights into generalizability

## Citation

If you use this code or the findings in your research, please cite:

```
@misc{emotion_recognition_vad,
  author = {AI Assistant},
  title = {Emotion Recognition using VAD Approach},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kookiemaster/emotion_recognition_iemocap}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Final Report: Investigation of Emotion Recognition on IEMOCAP

## Executive Summary

This report summarizes our investigation into the emotion recognition model using the IEMOCAP dataset. The investigation was prompted by concerns about suspiciously high accuracy, suggesting potential overfitting issues. Our analysis revealed that the high accuracy was not due to conventional overfitting but rather fundamental design flaws in the implementation. We identified the root causes, implemented proper cross-validation and parameter modifications, tested the model, and researched alternative approaches that properly implement text-to-VAD conversion for emotion recognition.

## Investigation Process

Our investigation followed a systematic approach:

1. **Repository Analysis**: We cloned and explored the repository structure to understand the codebase organization.

2. **Code Implementation Analysis**: We examined the main code files to understand the implementation details.

3. **VAD Tuple Approach Analysis**: We analyzed how the model converts text to Valence-Arousal-Dominance (VAD) tuples and then maps these to emotion categories.

4. **Model Architecture Examination**: We studied the model architecture to understand how it processes inputs and generates predictions.

5. **Data Preprocessing Investigation**: We investigated how the data is preprocessed before being fed into the model.

6. **Training Process Analysis**: We analyzed the training process to understand how the model learns from data.

7. **Performance Metrics Evaluation**: We evaluated the model's performance metrics to identify potential issues.

8. **Overfitting Analysis**: We created a comprehensive analysis script to check for signs of overfitting.

9. **Cross-Validation Implementation**: We implemented proper cross-validation to address potential overfitting issues.

10. **Parameter Modifications**: We modified model parameters to improve performance and reduce overfitting.

11. **Model Testing**: We tested the model with new parameters to evaluate their impact.

12. **Results Comparison**: We compared the results of the modified model with the original.

13. **Research on Alternative Approaches**: We researched similar approaches that properly implement text-to-VAD conversion.

14. **Documentation of Findings**: We documented our findings and recommendations for improvement.

## Key Findings

Our investigation revealed several critical issues with the emotion recognition model:

1. **No Actual Text Modality Used**: Despite the repository claiming to use text modality, no actual text data is processed. Instead, utterance IDs are used as placeholders:
   ```python
   # Since we don't have the actual text transcriptions, we'll use a pre-trained model approach
   # We'll use the utterance IDs as placeholders for now
   texts = vad_df['utterance_id'].tolist()
   ```

2. **Deterministic Rule-Based Emotion Mapping**: The VAD-to-emotion mapping is entirely rule-based with hard-coded thresholds:
   ```python
   if v >= midpoint and a >= midpoint:
       return 'happy'
   elif v >= midpoint and a < midpoint:
       return 'calm'
   ```

3. **Evaluation on the Entire Dataset**: The final accuracy is calculated on the entire dataset rather than a separate test set:
   ```python
   # Make predictions on the entire dataset
   predictions = classifier.predict(X)
   # Calculate accuracy
   accuracy = accuracy_score(y, predictions['predicted_emotion'])
   ```

4. **Data Leakage**: While the `train` method uses train_test_split, the final evaluation doesn't use this split, leading to data leakage.

5. **Perfect Separation by Design**: The VAD values perfectly separate the emotion classes by design, as emotions are deterministically derived from VAD values using rule-based approaches.

## Implementation and Testing

We implemented several modifications to address the identified issues:

1. **Proper Cross-Validation**: We implemented stratified k-fold cross-validation with proper train/validation/test splits (60%/20%/20%).

2. **Regularization Techniques**: We added L2 regularization, early stopping, class weights, and feature selection to prevent overfitting.

3. **Feature Engineering**: We implemented feature interactions, VAD ratios, and VAD differences to improve model performance.

4. **Multiple Classifier Types**: We tested Random Forest, SVM, and MLP classifiers with various parameter configurations.

Despite these modifications, the model still achieved perfect or near-perfect accuracy, confirming that the issue is not conventional overfitting but rather the fundamental design flaws identified earlier.

## Alternative Approaches

We researched alternative approaches that properly implement text-to-VAD conversion for emotion recognition:

1. **BERT-based Text-to-VAD Conversion**: We found a well-implemented approach in the [emotion-recognition-nlp-project](https://github.com/matanbt/emotion-recognition-nlp-project) repository that uses BERT for text processing and converting to VAD values.

2. **Two-Step Approach**: This approach first predicts VAD values from text using BERT embeddings and then maps these values to emotion categories.

3. **Proper Evaluation Methodology**: The researched approach implements proper evaluation methodology with train/validation/test splits, cross-validation, and multiple metrics.

## Recommendations

Based on our investigation and research, we recommend the following:

1. **Use Actual Text Data**: Implement the text-to-VAD conversion using real text transcriptions from the IEMOCAP dataset.

2. **Adopt BERT-based Approach**: Use a pre-trained BERT model to extract meaningful representations from text, followed by regression to predict VAD values.

3. **Implement Proper Evaluation**: Use proper train/validation/test splits with stratified sampling and cross-validation.

4. **Add Regularization**: Implement regularization techniques to prevent overfitting.

5. **Report Multiple Metrics**: Report F1 score, precision, recall, and confusion matrices in addition to accuracy.

6. **Cross-Dataset Validation**: Test the model on a different emotion dataset to evaluate generalization.

## Implementation Example

We provided a simplified example of how to implement a proper text-to-VAD conversion using BERT:

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class TextToVADModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vad_predictor = nn.Linear(config.hidden_size, 3)  # 3 for VAD dimensions
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, vad_targets=None):
        # Process text through BERT
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Predict VAD values
        vad_predictions = self.sigmoid(self.vad_predictor(pooled_output))
        
        # Calculate loss if targets provided
        loss = None
        if vad_targets is not None:
            loss_fn = nn.MSELoss()  # or nn.L1Loss() for MAE
            loss = loss_fn(vad_predictions, vad_targets)
            
        return loss, vad_predictions
```

## Conclusion

The current implementation does not actually perform emotion recognition from text as claimed. The perfect accuracy is an artifact of the circular process where emotions are deterministically derived from VAD values, and then models are trained to reproduce this mapping. To build a genuine emotion recognition system, the implementation needs to be significantly revised following our recommendations.

By implementing these changes, we can create a more realistic and useful emotion recognition system that actually learns to predict emotions from text data rather than simply reproducing deterministic rules.

## Deliverables

All our work has been committed to the GitHub repository:

1. **Analysis Scripts**: We created comprehensive analysis scripts to evaluate model performance and check for overfitting.

2. **Modified Models**: We implemented improved models with proper cross-validation and parameter modifications.

3. **Documentation**: We documented our findings and recommendations in detailed markdown files.

4. **Final Report**: This report summarizes our investigation, findings, and recommendations.

All changes have been pushed to the repository as requested.

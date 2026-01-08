# OCR-Driven-Deep-Learning-for-Non-Invasive-Thalassemia-Detection
This work presents a non-invasive, end-to-end thalassemia screening framework that converts CBC report images into diagnostic predictions by automatically extracting hematological parameters using OCR and classifying individuals as normal, silent carriers, or affected through a 1D CNN, eliminating the need for manual data entry or invasive testing.

Motivation

1. Thalassemia is a genetic blood disorder that requires early detection.

2. Traditional screening methods can be invasive, time-consuming, and resource-intensive.

3. Automating the process through deep learning and CBC report images can make screening faster, cheaper, and widely accessible.

4. Incorporating explainable AI ensures that predictions are transparent and clinically meaningful.

Dataset

The model is trained on a curated dataset of CBC blood parameters with corresponding thalassemia classification:
| Class | Description       | Number of Samples |
| ----- | ----------------- | ----------------- |
| 0     | Normal            | 2,500+            |
| 1     | Silent Carrier    | 3,000+            |
| 2     | Minor Thalassemia | 3,000+            |
| 3     | Major Thalassemia | 3,000+            |

Features used:

Age, Hb, Hct, MCV, MCH, MCHC, RDW, RBC count, Sex, RDW_Hb_ratio

Data is preprocessed and split into training and validation sets.

1. OCR-based Feature Extraction

Extract key blood parameters from CBC report images.

Convert image text into structured numerical data for model input.

2. 1D CNN Model

Input: 1D array of 11 blood features.

Architecture: Multiple convolutional layers with ReLU activation, followed by dense layers and softmax output for multi-class classification.

Loss: Categorical Crossentropy

Optimizer: Adam (learning rate = 0.001)

3. Model Training

Training data reshaped for CNN: (samples, features, 1)

Validation data used for performance evaluation.

Explainable AI

SHAP (SHapley Additive exPlanations)

1. Global feature importance and local patient-level explanations.

2. Key features influencing predictions: Hb, MCV, RDW, RDW_Hb_ratio.

3. Non-clinical features checked for bias.

Result:
Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.88      | 0.81   | 0.85     | 64      |
| 1     | 0.80      | 0.89   | 0.84     | 64      |
| 2     | 0.97      | 0.92   | 0.94     | 64      |
| 3     | 0.97      | 0.98   | 0.98     | 64      |
| **Accuracy** |           |        | 0.90     | 256     |
| **Macro Avg** | 0.91      | 0.90   | 0.90     | 256     |
| **Weighted Avg** | 0.91      | 0.90   | 0.90     | 256     |


#This work was completed for research purposes and has been accepted for presentation at ICDAM 2026 and publication in IEEE Xplore Digital Library.

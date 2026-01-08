import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import shap

# ----------------------
# Step 1: Load preprocessed data and trained model
train_df = pd.read_csv("train_preprocessed_novel.csv")
val_df = pd.read_csv("val_preprocessed_novel.csv")

X_train = train_df.drop(columns=["Group"]).values
y_train = train_df["Group"].values
X_val = val_df.drop(columns=["Group"]).values
y_val = val_df["Group"].values

# Reshape for CNN input
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

# Load the trained CNN model
model = load_model("thalassemia_1dcnn_model.h5")
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Feature names
feature_columns = [
    'Age', 'Hb', 'Hct', 'MCV', 'MCH', 'MCHC', 'RDW', 'RBC count',
    'Sex_female', 'Sex_male', 'RDW_Hb_ratio'
]

# ----------------------
# Step 2: Define wrapper for SHAP
def model_predict(X):
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
    return model.predict(X_reshaped, verbose=0)

# Use a subset of training data as background for SHAP
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
background_cnn = background.reshape(background.shape[0], background.shape[1], 1)

# ----------------------
# Step 3: Compute SHAP values
try:
    explainer = shap.DeepExplainer(model, background_cnn)
    shap_values = explainer.shap_values(X_val_cnn[:50])
except Exception as e:
    print(f"DeepExplainer failed: {e}")
    print("Falling back to KernelExplainer...")
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(X_val[:50], nsamples=50)  # Reduced nsamples for speed

# Handle SHAP values aggregation
if isinstance(shap_values, list):
    # Multi-class
    shap_values_agg = np.mean([np.abs(sv).squeeze(axis=-1) for sv in shap_values], axis=0)  # Shape: (50, 11)
else:
    # Single array
    shap_values_agg = np.abs(shap_values).mean(axis=3).squeeze(axis=2)  # Shape: (50, 11)

# Save SHAP values to CSV
shap_df = pd.DataFrame(shap_values_agg, columns=feature_columns)
shap_df.to_csv("shap_values_validation.csv", index=False)
print("SHAP values saved to 'shap_values_validation.csv'")

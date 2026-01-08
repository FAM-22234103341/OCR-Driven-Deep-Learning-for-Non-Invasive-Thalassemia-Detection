
df = pd.read_csv("/content/CBC dataset.csv") #Upload your dataset path 

# Step 1: Check & handle missing values
print("Missing values per column:")
print(df.isnull().sum())

# Step 2: One-hot encode 'Sex' instead of binary mapping
df = pd.get_dummies(df, columns=['Sex 1: female / 2: male'], drop_first=False)
# Now two columns: 'Sex 1: female / 2: male_1', 'Sex 1: female / 2: male_2'

# Step 3: Create new feature - RDW/Hb ratio (clinical insight)
df['RDW_Hb_ratio'] = df['RDW'] / df['Hb']

# Step 4: Extract features and labels
feature_columns = [
    'Age', 'Hb', 'Hct', 'MCV', 'MCH', 'MCHC', 'RDW', 'RBC count',
    'Sex 1: female / 2: male_1', 'Sex 1: female / 2: male_2',
    'RDW_Hb_ratio'
]
X = df[feature_columns].values
y = df['Group'].values - 1  # CNN requires zero-based class labels

# Step 5: Robust scaling (less sensitive to outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Handle class imbalance using SMOTE + TomekLinks
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_scaled, y)

# Step 7: Stratified train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Step 8: Reshape for 1D CNN input
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

# Step 9: Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# Step 10: Save preprocessed data
train_df = pd.DataFrame(X_train, columns=feature_columns)
train_df['Group'] = y_train
val_df = pd.DataFrame(X_val, columns=feature_columns)
val_df['Group'] = y_val

train_df.to_csv("train_preprocessed_novel.csv", index=False)
val_df.to_csv("val_preprocessed_novel.csv", index=False)

print("Novel preprocessing complete: 'train_preprocessed_novel.csv' and 'val_preprocessed_novel.csv'")

data = pd.read_csv("/content/train_preprocessed_novel.csv")

# Quick look
print(data.head())
print(data['Group'].value_counts())

data = pd.read_csv("/content/val_preprocessed_novel.csv")

# Quick look
print(data.head())
print(data['Group'].value_counts())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load preprocessed data
train_df = pd.read_csv("/content/train_preprocessed_novel.csv")
val_df = pd.read_csv("/content/val_preprocessed_novel.csv")
full_df = pd.concat([train_df, val_df], ignore_index=True)

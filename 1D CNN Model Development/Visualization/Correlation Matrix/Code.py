# 3. Correlation heatmap
plt.figure(figsize=(10,8))
corr = full_df.drop(columns='Group').corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

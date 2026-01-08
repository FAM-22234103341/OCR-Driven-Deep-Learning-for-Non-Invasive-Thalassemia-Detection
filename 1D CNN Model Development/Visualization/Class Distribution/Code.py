# 1. Class distribution
plt.figure(figsize=(6,4))
sns.countplot(data=full_df, x='Group', palette='Set2')
plt.title("Class Distribution (Group)")
plt.xlabel("Group (0: Normal, 1: α-thal-2, 2: α-thal-1, 3: HbH)")
plt.ylabel("Count")
plt.show()

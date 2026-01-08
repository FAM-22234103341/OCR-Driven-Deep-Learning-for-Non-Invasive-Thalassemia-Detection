# 2. Pairplot for some features
selected_features = ['Hb', 'MCV', 'MCH', 'RBC count']
sns.pairplot(full_df[selected_features + ['Group']], hue='Group', palette='tab10')
plt.suptitle("Pairplot of Key CBC Features by Class", y=1.02)
plt.show()

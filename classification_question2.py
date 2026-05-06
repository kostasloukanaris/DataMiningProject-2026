import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# load cleaned dataset from question 1
df = pd.read_csv("output_plots/dataset_cleaned.csv")

#separate features and target
X = df.drop(columns=["Label"]) #features (numerical all except Label)
y = df["Label"] #label only

#encode target labels to integers for classification
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("=" * 50)
print("Classes:")
for i, cls in enumerate(le.classes_):
    print(f"  {i}: {cls}")

#split: 70% train, 15% validation, 15% test
#x_train and y_train are now 70% of the originac cause we cut off 30%, x_temp and y_temp are the remaining 30% test set that i below cut into 15 for training and 15 for validation
#stratify=y_encoded means we want to maintain the same class distribution in each split as in the original dataset, if labelA is 60% of the original dataset, we want it to be 60% in train, val, and test as well
#also random_state=42 is the seed of the randomizer for the split (also row n of X stays together mapped with row n of y)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded) 

#then the 30% into 2
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print("\n" + "=" * 50)
print("Split sizes:")
print(f"  Train      : {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation : {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test       : {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

#scale features (required for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit only on train
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print("\n" + "=" * 50)
print("Scaling done (StandardScaler fit on train only)")

#check class imbalance in train set
unique, counts = np.unique(y_train, return_counts=True)
print("\n" + "=" * 50)
print("Class distribution in train set:")
for cls_id, cnt in zip(unique, counts):
    pct = cnt / len(y_train) * 100
    print(f"  {le.classes_[cls_id]:<40} {cnt:>8,}  ({pct:.2f}%)")

print("\nData preparation complete.")
print("  X_train_scaled, X_val_scaled, X_test_scaled are ready.")
print("  Use class_weight='balanced' in classifiers to handle imbalance.")
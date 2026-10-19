import pandas as pd
import numpy as np
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

OUTPUT_FOLDER = "./output_plots"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#load cleaned dataset produced by 01_eda.py
df = pd.read_csv("output_plots/dataset_cleaned.csv")

#separate features and target
X = df.drop(columns=["Label"])  # features (numerical all except Label)
y = df["Label"]                 # label only

#encode target labels to integers for classification
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("=" * 50)
print("Classes:")
for i, cls in enumerate(le.classes_):
    print(f"  {i}: {cls}")

#split: 70% train, 15% validation, 15% test
#x_train and y_train are now 70% of the original cause we cut off 30%, x_temp and y_temp are the remaining 30% test set that i below cut into 15 for training and 15 for validation
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


#helper
#takes a trained model, runs it on the test set, prints metrics and saves confusion matrix
#returns a dict with the results for the final comparison table

def evaluate(name, model, X_t, y_t, scenario):
    y_pred = model.predict(X_t)
    acc  = accuracy_score(y_t, y_pred)
    prec = precision_score(y_t, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_t, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_t, y_pred, average="weighted", zero_division=0)

    print(f"\n  [{scenario}] {name}")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}")
    print(f"    F1-score : {f1:.4f}")

    #save confusion matrix as png (rows=actual class, cols=predicted class)
    cm_norm = confusion_matrix(y_t, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Normalized Confusion Matrix - {name} ({scenario})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fname = f"cm_{name.replace(' ','_')}_{scenario}.png"
    plt.savefig(os.path.join(OUTPUT_FOLDER, fname), dpi=120)
    plt.close()

    return {"model": name, "scenario": scenario,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

results = []

#Decision Tree
#does not need scaled data (tree splits are scale-invariant)
#hyperparameters: max_depth (how deep the tree can grow), min_samples_split (minimum samples needed to split a node)

print("\n" + "=" * 50)
print("DECISION TREE")
print("=" * 50)

param_grid_dt = {
    'max_depth': [3, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

#Scenario A: manually loop all combinations, score each on validation set, pick best
print("\nScenario A: Grid Search with validation set...")
best_score_a = -1
best_params_a = None
for max_d in param_grid_dt['max_depth']:
    for min_s in param_grid_dt['min_samples_split']:
        dt = DecisionTreeClassifier(
            max_depth=max_d, min_samples_split=min_s,
            class_weight='balanced', random_state=42)
        dt.fit(X_train, y_train)
        # score on validation set (not test set)
        score = f1_score(y_val, dt.predict(X_val), average='weighted', zero_division=0)
        if score > best_score_a:
            best_score_a = score
            best_params_a = {'max_depth': max_d, 'min_samples_split': min_s}

print(f"  Best params: {best_params_a}  (val F1={best_score_a:.4f})")
# retrain on full train set with best params then evaluate on test set
dt_a = DecisionTreeClassifier(**best_params_a, class_weight='balanced', random_state=42)
dt_a.fit(X_train, y_train)
results.append(evaluate("Decision Tree", dt_a, X_test, y_test, "Scenario_A"))

#Scenario B: GridSearchCV handles all combinations + 3-fold CV automatically
#for each combination it trains 3 times (one per fold) and averages the scores
print("\nScenario B: Grid Search with 3-fold CV...")
gs_dt = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    param_grid_dt, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
gs_dt.fit(X_train, y_train)
print(f"  Best params: {gs_dt.best_params_}  (CV F1={gs_dt.best_score_:.4f})")

#best_estimator_ is already retrained on full train set by GridSearchCV
results.append(evaluate("Decision Tree", gs_dt.best_estimator_, X_test, y_test, "Scenario_B"))

#Random Forest
#like many decision trees, more robust than a single tree
#does not need scaled data
#hyperparameters: n_estimators (how many trees), max_depth (depth of each tree)
print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)

param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

#Scenario A
print("\nScenario A: Grid Search with validation set...")
best_score_a = -1
best_params_a = None
for n_est in param_grid_rf['n_estimators']:
    for max_d in param_grid_rf['max_depth']:
        rf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_d,
            class_weight='balanced', random_state=42, n_jobs=1)
        rf.fit(X_train, y_train)
        score = f1_score(y_val, rf.predict(X_val), average='weighted', zero_division=0)
        if score > best_score_a:
            best_score_a = score
            best_params_a = {'n_estimators': n_est, 'max_depth': max_d}

print(f"  Best params: {best_params_a}  (val F1={best_score_a:.4f})")
rf_a = RandomForestClassifier(**best_params_a, class_weight='balanced', random_state=42, n_jobs=1)
rf_a.fit(X_train, y_train)
results.append(evaluate("Random Forest", rf_a, X_test, y_test, "Scenario_A"))

#Scenario B
print("\nScenario B: Grid Search with 3-fold CV...")
gs_rf = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1),
    param_grid_rf, cv=3, scoring='f1_weighted', n_jobs=1, verbose=1)

gs_rf.fit(X_train, y_train)
print(f"  Best params: {gs_rf.best_params_}  (CV F1={gs_rf.best_score_:.4f})")
results.append(evaluate("Random Forest", gs_rf.best_estimator_, X_test, y_test, "Scenario_B"))

#Logistic Regression
#linear classifier, needs scaled data (sensitive to feature scale)
#hyperparameters: C (regularization strength, smaller = more regularization), max_iter (max training iterations)
print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'max_iter': [100, 500]
}

#Scenario A
print("\nScenario A: Grid Search with validation set...")
best_score_a = -1
best_params_a = None
for c in param_grid_lr['C']:
    for max_i in param_grid_lr['max_iter']:
        lr = LogisticRegression(
            C=c, max_iter=max_i,
            class_weight='balanced', random_state=42)
        lr.fit(X_train_scaled, y_train)
        score = f1_score(y_val, lr.predict(X_val_scaled), average='weighted', zero_division=0)
        if score > best_score_a:
            best_score_a = score
            best_params_a = {'C': c, 'max_iter': max_i}

print(f"  Best params: {best_params_a}  (val F1={best_score_a:.4f})")
lr_a = LogisticRegression(**best_params_a, class_weight='balanced', random_state=42)
lr_a.fit(X_train_scaled, y_train)
results.append(evaluate("Logistic Regression", lr_a, X_test_scaled, y_test, "Scenario_A"))

#Scenario B
print("\nScenario B: Grid Search with 3-fold CV...")
gs_lr = GridSearchCV(
    LogisticRegression(class_weight='balanced', random_state=42),
    param_grid_lr, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
gs_lr.fit(X_train_scaled, y_train)
print(f"  Best params: {gs_lr.best_params_}  (CV F1={gs_lr.best_score_:.4f})")
results.append(evaluate("Logistic Regression", gs_lr.best_estimator_, X_test_scaled, y_test, "Scenario_B"))

#FINAL COMPARISON
#all 6 combinations (3 classifiers x 2 scenarios)
print("\n" + "=" * 50)
print("FINAL COMPARISON")
print("=" * 50)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUTPUT_FOLDER, "classification_results.csv"), index=False)
print("\nSaved: output_plots/classification_results.csv")
print("Confusion matrix plots saved in output_plots/")

#bar chart comparing F1 across all 6 combinations
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(results_df))
bars = ax.bar(x, results_df["f1"], color=["steelblue","royalblue","seagreen","limegreen","salmon","tomato"])
ax.set_xticks(x)
ax.set_xticklabels([f"{r['model']}\n({r['scenario']})" for _, r in results_df.iterrows()], fontsize=8)
ax.set_ylabel("F1-score (weighted)")
ax.set_ylim(0.7, 1.01)
ax.set_title("F1-score Comparison - All Models and Scenarios")
for bar, val in zip(bars, results_df["f1"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f"{val:.4f}", ha="center", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "04_f1_comparison.png"), dpi=150)
plt.close()
print("saved: output_plots/04_f1_comparison.png")
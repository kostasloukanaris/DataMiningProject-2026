import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")
matplotlib.use("Agg")


DATA_FOLDER = "data/"       #the csv files must be here
OUTPUT_FOLDER = "output_plots/" #created if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("=" * 60)
print("step 1: loading data")
print("=" * 60)

csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
if not csv_files:
    raise FileNotFoundError(
        f"no csv files found in: {DATA_FOLDER}\n"
        "change DATA_FOLDER at the top of the file."
    )

frames = []
for f in csv_files:
    print(f"  loading {os.path.basename(f)}")
    tmp = pd.read_csv(f, encoding="utf-8", low_memory=False)
    frames.append(tmp)

#all 8 csvs become one big dataframe cause they have the same columns and its convenient to work with one df
df = pd.concat(frames, ignore_index=True)
print(f"\n total rows: {len(df):,}")
print(f" total cols: {df.shape[1]}")

#cleanup names
df.columns = df.columns.str.strip()


print("\n" + "=" * 60)
print("step 2: dataset structure")
print("=" * 60)

print(f"\nshape: {df.shape}")
print("\ndtypes per column:")
print(df.dtypes.to_string())



print("\n" + "=" * 60)
print("step 3: missing values & duplicates")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)

if missing_df.empty:
    print("\n no missing values, nice")
else:
    print(f"\n columns with missing values:\n{missing_df.to_string()}")
    df.dropna(inplace=True)
    print(f"  ->dropped NaN rows, shape now {df.shape}")

#inf values show up a lot in this dataset cause of division by zero in flow features
inf_mask = df.isin([np.inf, -np.inf])
inf_count = inf_mask.sum().sum()
print(f"\ninf values: {inf_count}")
if inf_count > 0:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"  ->replaced with NaN and dropped, shape now {df.shape}")

dups = df.duplicated().sum()
print(f"\nduplicate rows: {dups:,}")
if dups > 0:
    df.drop_duplicates(inplace=True)
    print(f"  ->removed, shape now {df.shape}")


#overflow values are always in the millions-to-billions negative, real traffic wont go below -1000
before = len(df)
numeric_cols = df.select_dtypes(include=[np.number]).columns
overflow_mask = (df[numeric_cols] < -1000).any(axis=1)
df = df[~overflow_mask]
dropped = before - len(df)
print(f"\noverflow rows dropped: {dropped} (shape now {df.shape})")


print("\n" + "=" * 60)
print("step 4: label distribution")
print("=" * 60)

label_counts = df["Label"].value_counts()
label_pct    = df["Label"].value_counts(normalize=True) * 100

print("\nclass breakdown:")
for label, cnt in label_counts.items():
    print(f"  {label:<40} {cnt:>8,}  ({label_pct[label]:.2f}%)")

fig, ax = plt.subplots(figsize=(12, 5))
label_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
ax.set_title("Κατανομή Κλάσεων (Label)", fontsize=14)
ax.set_xlabel("Κλάση")
ax.set_ylabel("Αριθμός Εγγραφών")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "01_label_distribution.png"), dpi=150)
plt.close()
print(f"\n saved: output_plots/01_label_distribution.png")

max_pct = label_pct.max()
min_pct = label_pct.min()
if max_pct > 80:
    print(f"\n heads up: '{label_pct.idxmax()}' is {max_pct:.1f}% of the data, pretty imbalanced")


print("\n" + "=" * 60)
print("step 5: basic stats")
print("=" * 60)

numeric_df = df.select_dtypes(include=[np.number])
stats = numeric_df.describe().T
stats["variance"] = numeric_df.var()

print("\nstats for first 10 columns, the other are in the csv file:")
print(stats.head(10).to_string())

stats.to_csv(os.path.join(OUTPUT_FOLDER, "statistics_summary.csv"))
print("\n saved: output_plots/statistics_summary.csv")


print("\n" + "=" * 60)
print("step 6: boxplots")
print("=" * 60)

#hardcoded key cols for the boxplot, cant plot all 80
box_cols = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Flow Bytes/s", "Flow Packets/s", "Fwd Packet Length Mean",
    "Bwd Packet Length Mean", "Flow IAT Mean", "Fwd IAT Mean", "Bwd IAT Mean"
]
box_cols = [c for c in box_cols if c in numeric_df.columns]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(box_cols[:10]):
    numeric_df[col].plot(kind="box", ax=axes[i], vert=True)
    axes[i].set_title(col, fontsize=8)
    axes[i].tick_params(axis="x", labelbottom=False)
plt.suptitle("Boxplots — Εντοπισμός Ακραίων Τιμών", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "02_boxplots.png"), dpi=150)
plt.close()
print(" saved: output_plots/02_boxplots.png")


print("\n" + "=" * 60)
print("step 7: correlation heatmap")
print("=" * 60)

#only 20 cols or the heatmap gets unreadable
sample_cols = numeric_df.columns[:20].tolist()
corr_matrix = numeric_df[sample_cols].corr()

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    corr_matrix,
    annot=False,
    cmap="coolwarm",
    center=0,
    linewidths=0.3,
    ax=ax
)
ax.set_title("Heatmap Συσχέτισης (πρώτες 20 στήλες)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "03_correlation_heatmap.png"), dpi=150)
plt.close()
print(" saved: output_plots/03_correlation_heatmap.png")


print("\n" + "=" * 60)
print("step 8: high correlation (>0.95)")
print("=" * 60)

corr_full = numeric_df.corr().abs()
#upper triangle only so we dont count each pair twice
upper = corr_full.where(np.triu(np.ones(corr_full.shape), k=1).astype(bool))
high_corr_pairs = [
    (col, row, upper.loc[row, col])
    for col in upper.columns
    for row in upper.index
    if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > 0.95
]

cols_to_drop_corr = set()
print(f"\npairs with correlation > 0.95 ({len(high_corr_pairs)} total):")
for c1, c2, val in high_corr_pairs[:20]:
    print(f"  {c1} <-> {c2}  (r={val:.3f})")
    cols_to_drop_corr.add(c2)  #drop c2, keep c1 (arbitrary)

print(f"\nwill drop {len(cols_to_drop_corr)} columns for high correlation")


print("\n" + "=" * 60)
print("step 9: low variance (VarianceThreshold)")
print("=" * 60)

#scale first or else features with huge ranges dominate the variance threshold
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numeric_scaled = pd.DataFrame(
    scaler.fit_transform(numeric_df.fillna(0)),
    columns=numeric_df.columns
)

selector = VarianceThreshold(threshold=0.01)  #drop anything below 1% variance
selector.fit(numeric_scaled)
low_var_cols = numeric_df.columns[~selector.get_support()].tolist()
print(f"low variance columns (<1%): {low_var_cols}")


print("\n" + "=" * 60)
print("step 10: feature selection — final dataset")
print("=" * 60)

cols_to_remove = set()

#high correlation
cols_to_remove.update(cols_to_drop_corr)

#low variance
cols_to_remove.update(low_var_cols)

#non-numeric cols (except Label which we need)
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
non_numeric = [c for c in non_numeric if c != "Label"]
cols_to_remove.update(non_numeric)

print(f"\ndropping {len(cols_to_remove)} columns total")
for reason, cols in [
    ("high correlation (>0.95)", cols_to_drop_corr),
    ("low variance",             set(low_var_cols)),
    ("non-numeric (except Label)", set(non_numeric)),
]:
    print(f"\n  [{reason}] ({len(cols)} cols):")
    for c in sorted(cols):
        print(f"    - {c}")

df_final = df.drop(columns=list(cols_to_remove), errors="ignore")
df_final.to_csv(os.path.join(OUTPUT_FOLDER, "dataset_cleaned.csv"), index=False)

print(f"\n started with : {df.shape[1]} cols")
print(f" ended with   : {df_final.shape[1]} cols")
print(f" rows         : {df_final.shape[0]:,}")
print(f" saved        : output_plots/dataset_cleaned.csv")
print("\n" + "=" * 60)
print("done — check output_plots/")
print("=" * 60)

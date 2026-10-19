import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

OUTPUT_FOLDER = "./output_plots"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# load cleaned dataset produced by 01_eda.py
#Label is kept only for evaluation after clustering, not using it now
DF_PATH = "output_plots/dataset_cleaned.csv"
df = pd.read_csv(DF_PATH)


X = df.drop(columns=["Label"])
y_label = df["Label"]

print("=" * 60)
print("Clustering")
print("=" * 60)
print(f"rows: {len(df):,}")
print(f"features: {X.shape[1]}")

#scale features (clustering is scale-sensitive)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#sample random for heavy clustering algorithms
SAMPLE_SIZE = 50000
if len(X_scaled) > SAMPLE_SIZE:
    rng = np.random.RandomState(67)
    idx = rng.choice(len(X_scaled), size=SAMPLE_SIZE, replace=False) #random sample without replacement, so its unique
    X_work = X_scaled[idx]
    y_work = y_label.iloc[idx].reset_index(drop=True) #match indexes with working X 
    print(f"using sample: {SAMPLE_SIZE:,} rows")
else:
    X_work = X_scaled
    y_work = y_label.reset_index(drop=True)
    print("using full dataset")

def cluster_scores(X_in, labels):
    unique = np.unique(labels)
    n_clusters = len(unique) - (1 if -1 in unique else 0) #for dbscan outliers
    noise_pct = 0.0
    if -1 in unique:
        noise_pct = (labels == -1).mean() * 100

    if n_clusters <= 1:
        silhouette = np.nan
        db_index = np.nan
    else:
        if -1 in unique:
            mask = labels != -1
            X_eval = X_in[mask]
            labels_eval = labels[mask]
        else:
            X_eval = X_in
            labels_eval = labels

        silhouette = silhouette_score(X_eval, labels_eval)
        db_index = davies_bouldin_score(X_eval, labels_eval)

    return n_clusters, noise_pct, silhouette, db_index


results = []


#K-Means
print("\n" + "=" * 60)
print("K-MEANS")
print("=" * 60)

k_values = [2, 3, 4, 5, 6, 8, 10]

k_sil = []
k_db = []

for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=67) #10 times with different centroid seeds, with the said k for each run
    labels = km.fit_predict(X_work) #trains ad returns the assigned cluster labels for each rown X_work
    n_clusters, noise_pct, sil, dbi = cluster_scores(X_work, labels)

    results.append({
        "algorithm": "KMeans",
        "params": f"k={k}",
        "n_clusters": n_clusters,
        "noise_pct": noise_pct,
        "silhouette": sil,
        "davies_bouldin": dbi,
        "ari_vs_label": np.nan,
        "nmi_vs_label": np.nan
    })

    k_sil.append(sil)
    k_db.append(dbi)

#plot metrics vs k
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_values, k_sil, marker="o", label="Silhouette")
ax.set_xlabel("k")
ax.set_ylabel("Silhouette")
ax.set_title("K-Means: Silhouette vs k")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "05_kmeans_silhouette.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_values, k_db, marker="o", color="tomato", label="Davies-Bouldin")
ax.set_xlabel("k")
ax.set_ylabel("Davies-Bouldin")
ax.set_title("K-Means: Davies-Bouldin vs k")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "06_kmeans_davies_bouldin.png"), dpi=150)
plt.close()

#pick best k by max silhouette
best_k_idx = int(np.nanargmax(k_sil)) #index of largest silhouette score
best_k = k_values[best_k_idx]
print(f"best k by silhouette: {best_k}")

km_best = KMeans(n_clusters=best_k, n_init=10, random_state=42)
km_labels = km_best.fit_predict(X_work)


# Hierarchical Clustering
print("\n" + "=" * 60)
print("HIERARCHICAL")
print("=" * 60)

linkages = ["ward", "complete", "average"] #rules for how to compute distance between cluster when desciding wehich two to merge
#ward: smallest increase in within‑cluster variance, complete: maximum distance between any pair of points, average: distance between clusters is the average distance between all pairs of points across the two clusters


for linkage in linkages:
    for k in k_values:
        if linkage == "ward" and k < 2: #ward needs at least 2 clusters, edge case for k=2, but we are testing k>=2 anyway, so yeah
            continue
        hc = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = hc.fit_predict(X_work)
        n_clusters, noise_pct, sil, dbi = cluster_scores(X_work, labels)

        results.append({
            "algorithm": "Hierarchical",
            "params": f"linkage={linkage}, k={k}",
            "n_clusters": n_clusters,
            "noise_pct": noise_pct,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "ari_vs_label": np.nan,
            "nmi_vs_label": np.nan
        })

#pick best hierarchical by max silhouette
hc_rows = [r for r in results if r["algorithm"] == "Hierarchical"]
hc_best = max(hc_rows, key=lambda r: r["silhouette"] if not np.isnan(r["silhouette"]) else -1) #for all dicts pick the one with the max silhouette score
print(f"best hierarchical by silhouette: {hc_best['params']}")

best_hc_linkage = hc_best["params"].split(",")[0].split("=")[1].strip()
best_hc_k = int(hc_best["params"].split("k=")[1])

hc_best_model = AgglomerativeClustering(n_clusters=best_hc_k, linkage=best_hc_linkage)
hc_labels = hc_best_model.fit_predict(X_work)


# DBSCAN
print("\n" + "=" * 60)
print("DBSCAN")
print("=" * 60)

eps_values = [0.3, 0.5, 0.7, 1.0] #radius for neighbours
min_samples_values = [5, 10, 20] #minimum number of points to form a cluster (in this radius)

for eps in eps_values:
    for min_s in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=min_s, n_jobs=-1)
        labels = db.fit_predict(X_work)
        n_clusters, noise_pct, sil, dbi = cluster_scores(X_work, labels)

        results.append({
            "algorithm": "DBSCAN",
            "params": f"eps={eps}, min_samples={min_s}",
            "n_clusters": n_clusters,
            "noise_pct": noise_pct,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "ari_vs_label": np.nan,
            "nmi_vs_label": np.nan
        })

#pick best DBSCAN by max silhouette (ignore NaN)
db_rows = [r for r in results if r["algorithm"] == "DBSCAN" and not np.isnan(r["silhouette"])]
if db_rows:
    db_best = max(db_rows, key=lambda r: r["silhouette"])
    print(f"best DBSCAN by silhouette: {db_best['params']}")
else:
    db_best = None
    print("no valid DBSCAN result (all noise or single cluster)")


#PCA plots for best models
print("\n" + "=" * 60)
print("PCA VISUALS")
print("=" * 60)

pca = PCA(n_components=2) #project to 2D for visualization 
X_pca = pca.fit_transform(X_work)

# K-Means plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, s=6, cmap="tab20", alpha=0.6)
ax.set_title(f"K-Means (k={best_k}) - PCA")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "07_kmeans_pca.png"), dpi=150)
plt.close()

# Hierarchical plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=hc_labels, s=6, cmap="tab20", alpha=0.6)
ax.set_title(f"Hierarchical ({best_hc_linkage}, k={best_hc_k}) - PCA")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "08_hierarchical_pca.png"), dpi=150)
plt.close()

# DBSCAN plot (only if valid)
if db_best is not None:
    best_eps = float(db_best["params"].split(",")[0].split("=")[1])
    best_min_s = int(db_best["params"].split("min_samples=")[1])
    db_model = DBSCAN(eps=best_eps, min_samples=best_min_s, n_jobs=-1)
    db_labels = db_model.fit_predict(X_work)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=db_labels, s=6, cmap="tab20", alpha=0.6)
    ax.set_title(f"DBSCAN (eps={best_eps}, min_samples={best_min_s}) - PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "09_dbscan_pca.png"), dpi=150)
    plt.close()


# Save results and label comparisons
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["algorithm", "silhouette"], ascending=[True, False])
print(results_df.to_string(index=False))

results_df.to_csv(os.path.join(OUTPUT_FOLDER, "clustering_results.csv"), index=False)
print("saved: output_plots/clustering_results.csv")

# crosstabs for best models (label vs cluster)
km_tab = pd.crosstab(y_work, km_labels, normalize="index")
km_tab.to_csv(os.path.join(OUTPUT_FOLDER, "kmeans_label_crosstab.csv"))
print("saved: output_plots/kmeans_label_crosstab.csv")

hc_tab = pd.crosstab(y_work, hc_labels, normalize="index")
hc_tab.to_csv(os.path.join(OUTPUT_FOLDER, "hierarchical_label_crosstab.csv"))
print("saved: output_plots/hierarchical_label_crosstab.csv")

if db_best is not None:
    db_tab = pd.crosstab(y_work, db_labels, normalize="index")
    db_tab.to_csv(os.path.join(OUTPUT_FOLDER, "dbscan_label_crosstab.csv"))
    print("saved: output_plots/dbscan_label_crosstab.csv")

print("\nDone. Check output_plots/ for charts and csvs.")

I adopt a two-stage design here. First, we allow the number of clusters (k) to vary across different time windows to examine whether segmentation structure is sensitive to the level of available information. Variation in optimal k indicates structural changes in segmentation granularity under different aggregation levels. 

If the optimised k values across all time windows are the same, we continue apply this k-value into the actual KMean clustering. If the optimised k values across all time windows are different, we will fix a common k to ensure fair comparison. It can isolate the effect of information availability of segmentation performance. 

## Choose the optimised K

We adopt a two-step selection procedure for determining the optimal number of clusters. First, we use the elbow criterion to filter out unreasonable k values. Then, within the remaining candidates, we select the k that maximises the silhouette score. 

```python
def select_k_common(results_df):
    df = results_df.sort_values("k").reset_index(drop=True)

    df["diff"] = df["inertia"].diff().abs()

    threshold = df["diff"].mean()
    elbow_candidates = df[df["diff"] >= threshold]["k"]

    candidates = df[df["k"].isin(elbow_candidates)]
    candidates = candidates[(candidates["k"] >= 3) & (candidates["k"] <= 10)]

    if candidates.empty:
        candidates = df[(df["k"] >= 3) & (df["k"] <= 10)]

    best_k = candidates.loc[candidates["silhouette"].idxmax(), "k"]

    return int(best_k)
```

Then we evaluates clustering structure across different values of k by computing both inertia and silhouette score. These metrics are then used as inputs for a subsequent selection procedure. 

```python
def evaluate_kmeans_k_structure(X, k_range=(2, 10)):
    results = []

    for k in range(k_range[0], k_range[1] + 1):

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = kmeans.fit_predict(X)

        inertia = kmeans.inertia_

        sil = silhouette_score(X, labels)

        results.append((k, inertia, sil))

    results_df = pd.DataFrame(
        results,
        columns=["k", "inertia", "silhouette"]
    )

    results_df = results_df.sort_values("k").reset_index(drop=True)

    return results_df
```

For each time window, we extract the optimal nulber of clusters and then  corresponding structural metrics, creating a standardised summary for cross-window comparison. 

```python
def summarise_k_structure(k_eval_df, best_k, window_label):

    best_row = k_eval_df[k_eval_df["k"] == best_k].iloc[0]

    return {
        "window": window_label,
        "best_k": int(best_k),
        "silhouette": float(best_row["silhouette"]),
        "inertia": float(best_row["inertia"])
    }
```
```python
aggregation_window_data = {
    "3M": consumption_3M,
    "6M": consumption_6M,
    "12M": consumption_12M
}
```

## Showing the best k value according to each time window

```python
k_structure_results = []

for window_label, raw_data in aggregation_window_data.items():

    # Step 1: feature prep
    data_processed, X_scaled, scaler = prepare_segmentation_features(raw_data)

    # Step 2: evaluate K
    k_eval_df = evaluate_kmeans_k_structure(X_scaled)

    # Step 3: select best K
    best_k = select_k_common(k_eval_df)

    # Step 4: summarise
    summary = summarise_k_structure(k_eval_df, best_k, window_label)
    k_structure_results.append(summary)

# Final output
k_structure_df = pd.DataFrame(k_structure_results)
k_structure_df
```

### K Selection Results

|Window   |	Best_k   |	Silhouette   |	Inertia   |
|---------|----------|---------------|------------|
|	3M   |	3   |	0.905996   |	203.332017   |
|	6M   |	4   |	0.823419   |	245.624416   |
|	12M   |	4   |	0.694231   |	692.999693   |

The optimal number of clusters varies across time windows. It suggests that the segmentation structure is sensitive to information avaiability. 

### Visualise the results

```python
def plot_k_diagnostics(df, window_name, best_k):
    fig, ax1 = plt.subplots()

    # Inertia (Elbow)
    ax1.plot(df["k"], df["inertia"], marker='o')
    ax1.axvline(best_k, linestyle='--')  # Mark the best k
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia")

    # Silhouette
    ax2 = ax1.twinx()
    ax2.plot(df["k"], df["silhouette"], marker='s')
    ax2.set_ylabel("Silhouette Score")

    plt.title(f"{window_name} - K Selection Diagnostics")
    plt.show()
```
```python
for window_label, raw_data in aggregation_window_data.items():

    data_processed, X_scaled, scaler = prepare_segmentation_features(raw_data)

    k_eval_df = evaluate_kmeans_k_structure(X_scaled)

    best_k = select_k_common(k_eval_df)

    plot_k_diagnostics(k_eval_df, window_label, best_k)

    summary = summarise_k_structure(k_eval_df, best_k, window_label)
    k_structure_results.append(summary)
```

For K Selection Diagnostics (3M)

![3M K Diagnostics](./Results%20Storage/3M_k_diagnostics.png)

For K Selection Diagnostics (6M)

![6M K Diagnostics](./Results%20Storage/6M_k_diagnostics.png)

For K Selection Diagnostics (12M)

![12M K Diagnostics](./Results%20Storage/12M_k_diagnostics.png)

## Choose the common best k value

The optimised k values across all time windows are different, therefore, we need to fix a common k value for fair comparison. 

### K can be selected from below list

```python
K_lsit = [3, 4, 5]
```

### Choose the common k using downstream predictive performance and structural diagnostics

Now we can evaluate each candidate k using downstream predictive performance and structural diagnostcics across all time windows. The final k is selected based on its consistency and robustness across different levels of information availability. 

```python
from sklearn.metrics import r2_score
from scipy.stats import ks_2samp

def preprocess(data):
    df = data.copy()

    spend_cols = [col for col in df.columns if "Spent" in col]
    txn_cols = [col for col in df.columns if "Transaction" in col]

    df["Total_Spent"] = df[spend_cols].sum(axis=1)
    df["Total_Txn"] = df[txn_cols].sum(axis=1)

    df["log_spend"] = np.log1p(df["Total_Spent"])
    df["log_txn"] = np.log1p(df["Total_Txn"])

    X = df[["log_spend", "log_txn"]]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled

def run_kmeans_analysis(data, k):
    df, X = preprocess(data)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    df["cluster"] = labels

    # ===== 1. Variance Explained (R²) =====
    overall_mean = df["Total_Spent"].mean()
    ss_total = np.sum((df["Total_Spent"] - overall_mean) ** 2)

    ss_within = 0
    for c in range(k):
        cluster_data = df[df["cluster"] == c]["Total_Spent"]
        ss_within += np.sum((cluster_data - cluster_data.mean()) ** 2)

    r2 = 1 - ss_within / ss_total

    # ===== 2. KS separation =====
    ks_scores = []
    for i in range(k):
        for j in range(i+1, k):
            d1 = df[df["cluster"] == i]["Total_Spent"]
            d2 = df[df["cluster"] == j]["Total_Spent"]

            if len(d1) > 0 and len(d2) > 0:
                ks = ks_2samp(d1, d2).statistic
                ks_scores.append(ks)

    ks_avg = np.mean(ks_scores) if ks_scores else 0

    return {
        "r2": r2,
        "ks": ks_avg
    }
```

### Show the results

```python
results = []

for K in [3, 4, 5]:
    for name, data in aggregation_window_data.items():

        metrics = run_kmeans_analysis(data, K)

        results.append({
            "K": K,
            "window": name,
            "R2": metrics["r2"],
            "KS": metrics["ks"]
        })

robustness_df = pd.DataFrame(results)
robustness_df
```

| K | Window | R²       | KS       |
|---|--------|----------|----------|
| 3 | 3M     | 0.675866 | 0.858801 |
| 4 | 3M     | 0.758762 | 0.897005 |
| 5 | 3M     | 0.792837 | 0.853656 |
| 3 | 6M     | 0.574404 | 0.883641 |
| 4 | 6M     | 0.648297 | 0.827856 |
| 5 | 6M     | 0.695382 | 0.870234 |
| 3 | 12M    | 0.524907 | 0.918518 |
| 4 | 12M    | 0.604071 | 0.873096 |
| 5 | 12M    | 0.632993 | 0.827875 |

Overall, higher k improves predictive performance, but structural seperation does not consistently improve. Here shows a trade-off between predictive power and structural interpretability. 

Therefore, we can choose k = 4 as the common k value since it has moderate predictive power and relatively strong structural interpretability. Also, given the severe sparsity in low-freuqency data, structural stability and interpretability are prioritised over purely predictive performance. 

## Start the KMeans Clustering, k = 4

```python
def run_kmeans_fixed_k(data, k=4):

    data_processed, X_scaled, scaler = prepare_segmentation_features(data)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    data_processed["cluster"] = labels

    centers = kmeans.cluster_centers_

    profile = data_processed.groupby("cluster").agg({
        "log_spend": "mean",
        "log_txn": "mean",
        "CustomerID": "count"
    }).rename(columns={"CustomerID": "size"}).reset_index()

    return data_processed, centers, profile
```

Keep the results of clustering: 

```python
results = {}

for name, data in aggregation_window_data.items():
    df_clustered, centers, profile = run_kmeans_fixed_k(data, k=4)

    results[name] = {
        "df": df_clustered,
        "centers": centers,
        "profile": profile
    }
```

Start to build the cluster profile across all time windows:

```python
def build_cluster_profile(results):
    profiles = []

    for w in ["12M", "6M", "3M"]:
        df = results[w]["df"]

        profile = df.groupby("cluster").agg({
            "log_txn": "mean",
            "log_spend": "mean"
        })

        profile.columns = [
            "log_txn_mean", "log_spend_mean"
        ]

        profile["txn_mean"] = np.exp(profile["log_txn_mean"]) - 1
        profile["spend_mean"] = np.exp(profile["log_spend_mean"]) - 1

        profile["window"] = w
        profile["cluster"] = profile.index

        profile = profile.sort_values("spend_mean", ascending=False)

        profiles.append(profile.reset_index(drop=True))

    profile_df = pd.concat(profiles, ignore_index=True)

    return profile_df
```

```python
profile_df = build_cluster_profile(results)
profile_df.sort_values(["window","cluster"])
```

The result is: 

| Window | Cluster | txn_mean | spend_mean |
|--------|---------|----------|------------|
| 12M    | 0       | 2.279146 | 236.599055 |
| 12M    | 1       | 0.000000 | 0.000000   |
| 12M    | 2       | 1.000000 | 99.548934  |
| 12M    | 3       | 4.811308 | 519.167189 |
| 6M     | 0       | 1.000000 | 102.678049 |
| 6M     | 1       | 0.000000 | 0.000000   |
| 6M     | 2       | 3.529052 | 364.595056 |
| 6M     | 3       | 2.000000 | 211.001721 |
| 3M     | 0       | 0.000000 | 0.000000   |
| 3M     | 1       | 1.000000 | 104.729196 |
| 3M     | 2       | 2.000000 | 200.658621 |
| 3M     | 3       | 3.458603 | 348.185598 |

Reorder and lable the clusters: 

```python
def relabel_clusters(profile_df):
    new_profiles = []

    for w in ["3M", "6M", "12M"]:
        df_w = profile_df[profile_df["window"] == w].copy()

        df_w = df_w.sort_values("txn_mean")

        df_w["cluster_rank"] = range(len(df_w))

        new_profiles.append(df_w)

    return pd.concat(new_profiles, ignore_index=True)
```
```python
profile_df = relabel_clusters(profile_df)
```
```python
labels = ["Segment 0", "Segment 1", "Segment 2", "Segment 3"]
profile_df["segment"] = profile_df["cluster"].map(lambda x: labels[x])
```
```python
profile_df
```

The result is: 

| Window | Cluster | txn_mean | spend_mean | cluster_rank | Segment   |
|--------|---------|----------|------------|--------------|-----------|
| 3M     | 0       | 0.000000 | 0.000000   | 0            | Segment 0 |
| 3M     | 1       | 1.000000 | 104.729196 | 1            | Segment 1 |
| 3M     | 2       | 2.000000 | 200.658621 | 2            | Segment 2 |
| 3M     | 3       | 3.458603 | 348.185598 | 3            | Segment 3 |
| 6M     | 1       | 0.000000 | 0.000000   | 0            | Segment 1 |
| 6M     | 0       | 1.000000 | 102.678049 | 1            | Segment 0 |
| 6M     | 3       | 2.000000 | 211.001721 | 2            | Segment 3 |
| 6M     | 2       | 3.529052 | 364.595056 | 3            | Segment 2 |
| 12M    | 1       | 0.000000 | 0.000000   | 0            | Segment 1 |
| 12M    | 2       | 1.000000 | 99.548934  | 1            | Segment 2 |
| 12M    | 0       | 2.279146 | 236.599055 | 2            | Segment 0 |
| 12M    | 3       | 4.811308 | 519.167189 | 3            | Segment 3 |

### Visualise for all customers across time windows

```python
def plot_all_customers(results, profile_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    windows = ["3M", "6M", "12M"]
    segment_order = ["Segment 0", "Segment 1", "Segment 2", "Segment 3"]

    palette = {
        "Segment 0": "#66c2a5",
        "Segment 1": "#fc8d62",
        "Segment 2": "#8da0cb",
        "Segment 3": "#e78ac3"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, w in enumerate(windows):

        df = results[w]["df"].copy()

        profile_w = profile_df[profile_df["window"] == w].copy()

        mapping = profile_w.set_index("cluster")["segment"].to_dict()
        df["segment"] = df["cluster"].map(mapping)

        # ======================
        # 🔵 CUSTOMER POINTS
        # ======================
        sns.scatterplot(
            data=df,
            x="txn",
            y="spend",
            hue="segment",
            hue_order=segment_order,
            palette=palette,
            s=20,
            alpha=0.5,
            ax=axes[i],
            legend=(i == 2)
        )

        # ======================
        # 🔴 CENTROIDS
        # ======================
        sns.scatterplot(
            data=profile_w,
            x="txn_mean",
            y="spend_mean",
            hue="segment",
            hue_order=segment_order,
            palette=palette,
            marker="X",
            s=150,
            edgecolor="black",
            linewidth=1,
            legend=False,
            ax=axes[i]
        )

        axes[i].set_title(f"{w} Window")
        axes[i].set_xlabel("Transaction Frequency")
        axes[i].set_ylabel("Spend")

    plt.suptitle("Customer Distribution Across Time Windows (K=4)", fontsize=14)
    plt.tight_layout()
    save_path = "customer_distribution_k4.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
```
![KMean_Customer_Distribution_K4](./Results%20Storage/KMean_Customer_Distribution_K4.png)

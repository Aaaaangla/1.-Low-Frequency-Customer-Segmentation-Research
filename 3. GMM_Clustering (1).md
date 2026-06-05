Same with the KMean Clustering. We find out the best k value for each time window first. If the best k values are all the same, we use this k value as the common k. If the best k values are different, we choose a common k value for all the time windows. 

For Data Preparation please refer to the 2. KMeans_Clustering.md

## Choose the optimised k value for each time windows

I evaluate GMM clustering using both hard and probabilistic representations. Hard labels derived via maximum posterior assignment are used to ensure comparability with KMeans in downstream evaluation, while probabilistic outputs are analysed through uncertainty and entropy to capture clustering ambiguity. Model selection is guided by BIC and an elbow-based filtering approach. 

```python
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def evaluate_gmm_k_structure(X, k_range=(2, 10)):
    results = []

    for k in range(k_range[0], k_range[1] + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=42
        )

        gmm.fit(X)

        # 🔹 HARD
        labels = gmm.predict(X)
        sil = silhouette_score(X, labels)

        # 🔹 SOFT
        probs = gmm.predict_proba(X)

        # uncertainty
        confidence = probs.max(axis=1)
        uncertainty = 1 - confidence
        avg_uncertainty = uncertainty.mean()

        # entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        avg_entropy = entropy.mean()

        # BIC
        bic = gmm.bic(X)

        results.append((k, bic, sil, avg_uncertainty, avg_entropy))

    df = pd.DataFrame(
        results,
        columns=["k", "bic", "silhouette", "uncertainty", "entropy"]
    )

    df = df.sort_values("k").reset_index(drop=True)

    # elbow
    df["diff"] = df["bic"].diff().abs()

    return df
```
For choosing the best k, we aim to find the one that with the highest silhouette and lowest uncertainty. 

First, candidate k value are filtered using an elbow criterion based on the first-order differences in BIC, retaining only those associated with significant structural changes. 

Second, among the remaining candidates, we define a composite score that balances clustering separation and assignment certainty. Specifically, we maximise the difference between the silhouette score and the average uncertainty, favouring cluster configurations that are both well-separated and probabilistically confident. 

```python
def select_k_gmm(results_df):
    df = results_df.copy()

    # Step 1: elbow filter (based on BIC)
    threshold = df["diff"].mean()
    elbow_candidates = df[df["diff"] >= threshold]["k"]

    candidates = df[df["k"].isin(elbow_candidates)]
    candidates = candidates[(candidates["k"] >= 3) & (candidates["k"] <= 10)]

    # fallback
    if candidates.empty:
        candidates = df[(df["k"] >= 3) & (df["k"] <= 10)]

    candidates["score"] = (
        candidates["silhouette"] - candidates["uncertainty"]
    )

    best_k = candidates.loc[candidates["score"].idxmax(), "k"]

    return int(best_k)
```

### GMM Best k Results

```python
gmm_k_results = []

for window_label, raw_data in aggregation_window_data.items():

    data_processed, X_scaled, scaler = prepare_segmentation_features(raw_data)

    gmm_eval_df = evaluate_gmm_k_structure(X_scaled)

    best_k = select_k_gmm(gmm_eval_df)

    best_row = gmm_eval_df[gmm_eval_df["k"] == best_k].iloc[0]

    gmm_k_results.append({
        "window": window_label,
        "best_k": best_k,
        "silhouette": float(best_row["silhouette"]),
        "bic": float(best_row["bic"]),
        "uncertainty": float(best_row["uncertainty"]),
        "entropy": float(best_row["entropy"])
    })

gmm_k_df = pd.DataFrame(gmm_k_results)
gmm_k_df
```
| Time Window | Optimal k | Silhouette Score | BIC            | Avg. Uncertainty | Avg. Entropy |
|-------------|-----------|------------------|----------------|------------------|--------------|
| 3M          | 3         | 0.905996         | -102570.016616 | 0.000004         | 0.000050     |
| 6M          | 4         | 0.823419         | -82569.190665  | 0.000022         | 0.000212     |
| 12M         | 3         | 0.686461         | -37285.669584  | 0.000057         | 0.000567     |

### Visualise the results

```python
def plot_gmm_k_diagnostics(gmm_eval_df, best_k, window_label):
    fig, ax1 = plt.subplots(figsize=(8,5))

    k_vals = gmm_eval_df["k"]

    # Uncertainty
    ax1.plot(k_vals, gmm_eval_df["uncertainty"], marker='o', label="Uncertainty")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Uncertainty")
    ax1.set_title(f"{window_label} - GMM K Selection Diagnostics")

    # Silhouette
    ax2 = ax1.twinx()
    ax2.plot(k_vals, gmm_eval_df["silhouette"], marker='s', linestyle='--', label="Silhouette")
    ax2.set_ylabel("Silhouette Score")

    # Mark the best k
    ax1.axvline(x=best_k, linestyle='--')

    # legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    filename = f"GMM_K_diagnostics_{window_label}.png"
    print("Saving file:", filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()
```

For K Selection Diagnostics (3M)

![3M K Diagnostics](./Results%20Storage/GMM_K_diagnostics_3M.png)

For K Selection Diagnostics (6M)

![6M K Diagnostics](./Results%20Storage/GMM_K_diagnostics_6M.png)

For K Selection Diagnostics (12M)

![12M K Diagnostics](./Results%20Storage/GMM_K_diagnostics_12M.png)

The optimised k value in each time window is different. Therefore, we need to choose a common k value as we did in the KMeans Clustering as well. 

### Apply the k value list to each time window

```python
k_list = [3,4,5]
```
```python
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp
import numpy as np

def run_gmm_analysis(data, k):
    df, X = preprocess(data)

    gmm = GaussianMixture(n_components=k, random_state=42)
    probs = gmm.fit_predict(X) 

    df["cluster"] = probs

    # ===== 1. R² =====
    overall_mean = df["Total_Spent"].mean()
    ss_total = np.sum((df["Total_Spent"] - overall_mean) ** 2)

    ss_within = 0
    for c in range(k):
        cluster_data = df[df["cluster"] == c]["Total_Spent"]
        ss_within += np.sum((cluster_data - cluster_data.mean()) ** 2)

    r2 = 1 - ss_within / ss_total

    # ===== 2. KS =====
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
Show the results: 
```python
results = []

for K in [3,4,5]:
    for name, data in aggregation_window_data.items():

        metrics = run_gmm_analysis(data, K)

        results.append({
            "K": K,
            "window": name,
            "R2": metrics["r2"],
            "KS": metrics["ks"]
        })

gmm_robustness_df = pd.DataFrame(results)
gmm_robustness_df
```
| Number of Clusters (K) | Time Window | R² (Explained Variance) | KS Statistic |
|------------------------|-------------|--------------------------|--------------|
| 3                      | 3M          | 0.675866                 | 0.858801     |
| 4                      | 3M          | 0.752637                 | 0.904300     |
| 5                      | 3M          | 0.786712                 | 0.860073     |
| 3                      | 6M          | 0.574404                 | 0.883641     |
| 4                      | 6M          | 0.648297                 | 0.827856     |
| 5                      | 6M          | 0.692703                 | 0.874211     |
| 3                      | 12M         | 0.414029                 | 0.900560     |
| 4                      | 12M         | 0.545528                 | 0.834095     |
| 5                      | 12M         | 0.629987                 | 0.827692     |

The results reveal a clear trade-off between predictive performance and structural separation across different values of k and time windows. 

The R² score consistenly increase as the number of clusters increases across all time windows. However, the behaviour of the KS statistic is more nuanced and does not follow a monotonic patter. For the 3 month window, it has relatively the highest value when k = 4. In contrast, for the 6 month window, KS has relatively the lowest value when k = 4. And for the 12 month window, the highest KS occurs at k = 3, where R² is the lowest. 

Based on the above results, k = 4 is selected as the common segmentation level. While k = 3 yields the highest KS value in the 12 month window, it exhibits substantially lower R² across all windows, showing insufficient explanatory power. Conversely, k = 5 achieves the highest R², but does not consistently improves KS. It indicates potential over-segmentation and reduced structural coherence. Whereas, k = 4 provides a balanced trade-off between predictive performance and structural separation. It achieves relatively high R² across all time windows while maintaining stable and, in some cases, optimal KS values. 

## Apply GMM to all time windows with k value k = 4

In this case, two types of cluster profiles are constructed for GMM: 
* Hard Profile,
* Soft Profile

The **hard profile** assigns each observation to the cluster with the highest posterior probability, allowing direct comparison with KMeans and enabling structural and predictive evaluation. 

The **soft profile** uses probabilistic memberships to compute weighted cluster characteristics, capturing uncertainty and overlap between clusters. 

Together, these two profiles provides complementary insights: 
* Hard profile reflects structural clarity
* Soft profile reveals probabilistic ambuiguity

### For hard profile

```python
from sklearn.mixture import GaussianMixture

def run_gmm_fixed_k(data, k=4):

    data_processed, X_scaled, scaler = prepare_segmentation_features(data)

    gmm = GaussianMixture(n_components=k, random_state=42)
    
    labels = gmm.fit_predict(X_scaled)

    data_processed["cluster"] = labels

    # cluster centers
    centers = gmm.means_

    profile = data_processed.groupby("cluster").agg({
        "log_spend": "mean",
        "log_txn": "mean",
        "CustomerID": "count"
    }).rename(columns={"CustomerID": "size"}).reset_index()

    return data_processed, centers, profile
```
```python
gmm_results = {}

for name, data in aggregation_window_data.items():
    df_clustered, centers, profile = run_gmm_fixed_k(data, k=4)

    gmm_results[name] = {
        "df": df_clustered,
        "centers": centers,
        "profile": profile
    }
```
```python
gmm_profile_df = build_cluster_profile(gmm_results)
```
```python
def relabel_clusters(profile_df):
    new_profiles = []

    for w in ["3M", "6M", "12M"]:
        df_w = profile_df[profile_df["window"] == w].copy()

        df_w = df_w.sort_values("spend_mean")

        # 重新编号
        df_w["cluster_rank"] = range(len(df_w))

        new_profiles.append(df_w)

    return pd.concat(new_profiles, ignore_index=True)
```
```python
gmm_profile_df = relabel_clusters(gmm_profile_df)

labels = ["Segment 0", "Segment 1", "Segment 2", "Segment 3"]

gmm_profile_df["segment"] = gmm_profile_df["cluster_rank"].map(lambda x: labels[x])

gmm_profile_df = gmm_profile_df.sort_values(["window", "cluster_rank"]).reset_index(drop=True)

gmm_profile_df
```
The profile is: 

| Window | Cluster | txn_mean | spend_mean | cluster_rank | Segment   |
|--------|--------|----------|------------|--------------|-----------|
| 12M    | 1      | 0.000000 | 0.000000   | 0            | Segment 0 |
| 12M    | 3      | 1.000000 | 99.548934  | 1            | Segment 1 |
| 12M    | 0      | 2.000000 | 210.156388 | 2            | Segment 2 |
| 12M    | 2      | 3.785818 | 395.958163 | 3            | Segment 3 |
| 6M     | 1      | 0.000000 | 0.000000   | 0            | Segment 0 |
| 6M     | 0      | 1.000000 | 102.678049 | 1            | Segment 1 |
| 6M     | 3      | 2.000000 | 211.001721 | 2            | Segment 2 |
| 6M     | 2      | 3.529052 | 364.595056 | 3            | Segment 3 |
| 3M     | 1      | 0.000000 | 0.000000   | 0            | Segment 0 |
| 3M     | 0      | 1.000000 | 65.556316  | 1            | Segment 1 |
| 3M     | 3      | 1.000000 | 146.327537 | 2            | Segment 2 |
| 3M     | 2      | 2.225238 | 221.937109 | 3            | Segment 3 |

#### The visualisation:

```python
def plot_all_customers_hard(results, profile_df):
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

        # mapping
        mapping = profile_w.set_index("cluster")["segment"].to_dict()
        df["segment"] = df["cluster"].map(mapping)

        # 🔵 customers
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

        # 🔴 centroids
        sns.scatterplot(
            data=profile_w,
            x="txn_mean",
            y="spend_mean",
            hue="segment",
            palette=palette,
            marker="X",
            s=150,
            edgecolor="black",
            legend=False,
            ax=axes[i]
        )

        axes[i].set_title(f"{w} (Hard)")
        axes[i].set_xlabel("Transaction Frequency")
        axes[i].set_ylabel("Spend")

    plt.suptitle("GMM Hard Clustering (k=4)")
    plt.tight_layout()
    plt.savefig("gmm_hard_profile.png", dpi=300, bbox_inches="tight")
    plt.show()
```

The scatter plot for all customers:

![GMM Hard Profile Scatter Plot](./Results%20Storage/GMM_hard_profile.png)

### For Soft Profile: 

```python
from sklearn.mixture import GaussianMixture

def run_gmm_soft_profile(data, k=4):

    data_processed, X_scaled, scaler = prepare_segmentation_features(data)

    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)

    probs = gmm.predict_proba(X_scaled)   # shape: (n_samples, k)

    df = data_processed.copy()

    # ===== Soft profile =====
    soft_profiles = []

    for cluster_id in range(k):

        weights = probs[:, cluster_id]

        if weights.sum() == 0:
            continue

        log_txn_mean = np.sum(weights * df["log_txn"]) / np.sum(weights)
        log_spend_mean = np.sum(weights * df["log_spend"]) / np.sum(weights)

        txn_mean = np.exp(log_txn_mean) - 1
        spend_mean = np.exp(log_spend_mean) - 1

        size = weights.sum()  # soft cluster size

        soft_profiles.append({
            "cluster": cluster_id,
            "log_txn_mean": log_txn_mean,
            "log_spend_mean": log_spend_mean,
            "txn_mean": txn_mean,
            "spend_mean": spend_mean,
            "size": size
        })

    profile_df = pd.DataFrame(soft_profiles)

    return df, probs, profile_df
```
```python
gmm_soft_results = {}

for name, data in aggregation_window_data.items():
    df, probs, profile = run_gmm_soft_profile(data, k=4)

    profile["window"] = name

    gmm_soft_results[name] = {
        "df": df,
        "probs": probs,
        "profile": profile
    }
```
```python
soft_profile_df = pd.concat(
    [gmm_soft_results[w]["profile"] for w in ["12M", "6M", "3M"]],
    ignore_index=True
)
```
```python
soft_profile_df = soft_profile_df.sort_values(
    ["window", "txn_mean"]
)

soft_profile_df["cluster_rank"] = soft_profile_df.groupby("window").cumcount()

labels = ["Segment 0", "Segment 1", "Segment 2", "Segment 3"]
soft_profile_df["segment"] = soft_profile_df["cluster_rank"].map(lambda x: labels[x])
```
```python
soft_profile_df
```
The soft profile is: 

| Window | Cluster | txn_mean | spend_mean | size       | cluster_rank | Segment   |
|--------|--------|----------|------------|------------|--------------|-----------|
| 12M    | 1      | 0.000000 | 0.000000   | 686.000000 | 0            | Segment 0 |
| 12M    | 3      | 1.000000 | 99.548899  | 3202.996691| 1            | Segment 1 |
| 12M    | 0      | 2.000000 | 210.153221 | 1348.660070| 2            | Segment 2 |
| 12M    | 2      | 3.785152 | 395.890626 | 1162.343239| 3            | Segment 3 |
| 6M     | 1      | 0.000000 | 0.000000   | 2083.000000| 0            | Segment 0 |
| 6M     | 0      | 1.000000 | 102.678040 | 2937.999529| 1            | Segment 1 |
| 6M     | 3      | 2.000000 | 210.997007 | 940.861464 | 2            | Segment 2 |
| 6M     | 2      | 3.528458 | 364.549234 | 438.139007 | 3            | Segment 3 |
| 3M     | 1      | 0.000000 | 0.000000   | 3587.000000| 0            | Segment 0 |
| 3M     | 0      | 1.000000 | 70.385922  | 975.486837 | 1            | Segment 1 |
| 3M     | 3      | 1.000000 | 140.074839 | 1328.486409| 2            | Segment 2 |
| 3M     | 2      | 2.225157 | 221.930149 | 509.026754 | 3            | Segment 3 |

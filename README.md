# Customer Segmentation Under Information Constraints

## Overview

Customer segmentation remains one of the most widely used techniques in customer analytics, supporting marketing strategy, customer retention, resource allocation, and personalised customer engagement.

In practice, many organisations continue to rely on traditional RFM (Recency–Frequency–Monetary) frameworks to classify customers. While these approaches are simple and interpretable, they were originally developed under the assumption that sufficient customer behavioural information is available.

However, this assumption is often violated in low-frequency transaction environments such as utilities, insurance, real estate, automotive services, and many B2B industries. In these settings, customer interactions are sparse, irregular, and highly information-constrained.

This research investigates a fundamental question:

**Can meaningful customer segmentation still be achieved when historical customer information is severely limited?**

---

## Research Motivation

Most segmentation research focuses on developing new clustering algorithms or improving predictive performance.

This study takes a different perspective.

Rather than asking:

*"Which clustering algorithm performs best?"*

This research asks:

*"Do the fundamental assumptions behind customer segmentation remain valid under low-frequency transaction conditions?"*

If customer behaviour becomes too sparse, segmentation quality may deteriorate regardless of the algorithm being used.

Understanding this limitation is important because inaccurate customer segments can lead to ineffective targeting strategies, poor business decisions, and misleading customer insights.

---

## Research Design

This study evaluates customer segmentation under three levels of information availability:

| Information Level   | Historical Window |
| ------------------- | ----------------- |
| Limited Information | 3 Months          |
| Medium Information  | 6 Months          |
| Rich Information    | 12 Months         |

Customer behaviour is first transformed into RFM-based features and then segmented using two widely adopted AI-driven clustering approaches:

* K-Means Clustering
* Gaussian Mixture Models (GMM)

These methods are intentionally selected as industry-standard baselines rather than novel algorithms.

If segmentation quality consistently deteriorates across both methods, the problem is likely caused by information scarcity rather than algorithm selection.

---

## Evaluation Framework

To assess segmentation effectiveness, a multi-dimensional evaluation framework is proposed.

### Predictive Performance

Evaluates whether customer segments improve future behavioural prediction.

Metrics:

* R²
* RMSE

Prediction Models:

* OLS Regression
* XGBoost

### Structural Stability

Evaluates whether segmentation structures remain consistent as available information changes.

Metrics:

* Kolmogorov–Smirnov Distance
* Wasserstein Distance (W1)
* Distribution Overlap
* Cluster Size Consistency

### Structural Interpretability

Evaluates whether generated segments remain behaviourally meaningful and actionable.

Metrics:

* Monotonicity
* Variance Explained
* Rank Correlation
* Mutual Information
* Regression Fit (R²)

---

## Key Findings

Preliminary results indicate that customer segmentation effectiveness is highly sensitive to information availability.

As historical information decreases:

* Customer behaviour becomes increasingly sparse
* Segment boundaries become less distinguishable
* Overlap between customer groups increases
* Structural stability deteriorates
* Predictive usefulness declines

These findings suggest that segmentation quality may be driven more by information availability than by clustering algorithm choice.

---

## Research Contribution

This research contributes to customer analytics literature in three ways:

1. Highlights the limitations of applying traditional segmentation approaches in low-frequency transaction environments.

2. Proposes a practical evaluation framework for assessing segmentation effectiveness through prediction, stability, and interpretability.

3. Provides empirical evidence that information scarcity may be a more critical constraint than algorithm selection when performing customer segmentation.

---

## Technologies

Python • Scikit-Learn • Pandas • NumPy • K-Means • Gaussian Mixture Models • OLS Regression • XGBoost • Customer Analytics

---

## Research Status

Current Stage: Conference Paper Development

Future work will focus on developing segmentation methodologies specifically designed for low-frequency customer environments and validating the framework across additional industry datasets.

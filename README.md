# Low-Frequency Customer Segmentation Research

## Research Motivation

Customer segmentation is one of the most widely adopted techniques in customer analytics, supporting marketing strategy, customer retention, personalisation, and resource allocation decisions.

In practice, many organisations continue to rely on traditional RFM (Recency–Frequency–Monetary) segmentation frameworks due to their simplicity and interpretability. However, these approaches were originally developed under the assumption that customer transaction histories are sufficiently frequent and information-rich.

This assumption becomes problematic in many real-world industries where customer purchases occur infrequently. Examples include insurance, real estate, automotive sales, industrial equipment, utilities, and high-value B2B services. In these environments, customer behaviour is often characterised by sparse transactions, limited observations, and long periods of inactivity.

Under such conditions, it remains unclear whether conventional segmentation approaches can still generate meaningful customer groups.

---

## Research Problem

Most existing customer segmentation studies focus on improving clustering algorithms.

This research takes a different perspective.

Rather than asking:

*"Which segmentation algorithm performs best?"*

this study asks:

*"Can meaningful customer segmentation be achieved at all when behavioural information is severely limited?"*

The objective is therefore not to propose a new clustering method, but to investigate the structural limitations of customer segmentation under low-frequency transaction environments.

---

## Research Approach

To evaluate this problem, customer segments are generated using two widely adopted AI-driven clustering approaches:

* K-Means Clustering
* Gaussian Mixture Models (GMM)

Both methods are applied to RFM-derived behavioural features under different levels of historical information availability:

| Information Level   | Observation Window |
| ------------------- | ------------------ |
| Limited Information | 3 Months           |
| Medium Information  | 6 Months           |
| Rich Information    | 12 Months          |

These models serve as diagnostic tools rather than optimisation targets.

If segmentation quality deteriorates consistently across fundamentally different clustering paradigms, the problem is likely rooted in the underlying data structure rather than the choice of algorithm.

---

## Evaluation Framework

A multi-dimensional evaluation framework is developed to assess segmentation effectiveness from three complementary perspectives:

### 1. Predictive Performance

Can customer segments improve future behavioural prediction?

Metrics:

* R²
* RMSE

Models:

* OLS Regression
* XGBoost

### 2. Structural Stability

Do segmentation structures remain consistent as information availability changes?

Metrics:

* Kolmogorov–Smirnov Distance
* Wasserstein Distance (W1)
* Distribution Overlap
* Cluster Size Consistency

### 3. Structural Interpretability

Do generated segments remain behaviourally meaningful?

Metrics:

* Monotonicity
* Variance Explained
* Rank Correlation
* Mutual Information
* Regression Fit (R²)

---

## Preliminary Findings

Initial results suggest that segmentation effectiveness is highly sensitive to information availability.

As observation windows become shorter:

* Customer behaviour becomes increasingly sparse
* Segment boundaries become less distinguishable
* Distribution overlap between clusters increases
* Structural stability deteriorates
* Predictive usefulness declines

The findings indicate that low-frequency customer environments introduce fundamental challenges that may not be solvable solely through improved clustering algorithms.

Instead, the core issue may lie in the mismatch between traditional segmentation assumptions and the statistical characteristics of sparse behavioural data.

---

## Research Contribution

This study contributes to customer analytics literature in three ways:

1. Highlights the limitations of traditional segmentation assumptions in low-frequency industries.

2. Proposes a segmentation effectiveness evaluation framework incorporating prediction, stability, and interpretability.

3. Provides empirical evidence that data sparsity may be a more critical constraint than algorithm selection when performing customer segmentation.

This research forms the first stage of a broader research agenda focused on developing segmentation methodologies specifically designed for low-frequency customer environments.

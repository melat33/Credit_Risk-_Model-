Proxy Default Variable Creation

Dataset contains no repayment outcomes.

We engineer “is_high_risk” using:

✔ RFM Features

Recency → last activity date

Frequency → number of transactions

Monetary → total amount spent

✔ K-Means Clustering

Assign customers to 3 clusters:

Cluster 0 → High engagement → low risk

Cluster 1 → Medium engagement → medium risk

Cluster 2 → Low engagement → high risk (proxy = 1)

This becomes our PD target for Task-2 model training.
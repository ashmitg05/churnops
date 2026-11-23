| family | imbalance | val ROC-AUC | val PR-AUC | val F1 | val Brier | threshold | val cost/cust | MLflow run |
|---|---|---|---|---|---|---|---|---|
| logreg | class_weight | 0.8352 | 0.6439 | 0.5206 | 0.1659 | 0.13 | 40.40 | `4b634946` |
| logreg | smote | 0.8344 | 0.6434 | 0.5399 | 0.1641 | 0.16 | 40.11 | `2d3396be` |
| random_forest | smote | 0.8317 | 0.6202 | 0.5093 | 0.1526 | 0.09 | 40.92 | `5e7c9a45` |
| random_forest | class_weight | 0.8303 | 0.6267 | 0.5131 | 0.1601 | 0.12 | 40.54 | `723eedbf` |
| xgboost | smote | 0.8295 | 0.6286 | 0.5161 | 0.1435 | 0.04 | 40.54 | `341f6104` |
| lightgbm | class_weight | 0.8244 | 0.6165 | 0.5524 | 0.1618 | 0.11 | 39.36 | `b46f99ad` |
| lightgbm | smote | 0.8239 | 0.6176 | 0.5303 | 0.1463 | 0.05 | 40.96 | `a46e8f83` |
| xgboost | class_weight | 0.8231 | 0.6167 | 0.5489 | 0.1631 | 0.12 | 40.26 | `4bd97300` |

Tuned candidate (logreg/class_weight, run `79b5ce4f`) on the **test** set:

| ROC-AUC | PR-AUC | precision | recall | F1 | Brier | threshold | cost/cust |
|---|---|---|---|---|---|---|---|
| 0.8541 | 0.6683 | 0.3596 | 0.9929 | 0.5280 | 0.1627 | 0.13 | 37.51 |

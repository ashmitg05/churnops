| family | imbalance | val ROC-AUC | val PR-AUC | val F1 | val Brier | threshold | val cost/cust | MLflow run |
|---|---|---|---|---|---|---|---|---|
| logreg | class_weight | 0.8352 | 0.6439 | 0.5206 | 0.1659 | 0.13 | 40.40 | `c082a233` |
| logreg | smote | 0.8344 | 0.6435 | 0.5399 | 0.1641 | 0.16 | 40.11 | `0beb01f4` |
| random_forest | smote | 0.8317 | 0.6200 | 0.5151 | 0.1526 | 0.10 | 40.63 | `4c2a844a` |
| xgboost | smote | 0.8306 | 0.6346 | 0.5646 | 0.1427 | 0.10 | 40.21 | `9eda1208` |
| random_forest | class_weight | 0.8303 | 0.6267 | 0.5131 | 0.1601 | 0.12 | 40.54 | `675fbff0` |
| lightgbm | class_weight | 0.8244 | 0.6165 | 0.5524 | 0.1618 | 0.11 | 39.36 | `27c05c84` |
| lightgbm | smote | 0.8234 | 0.6136 | 0.5296 | 0.1468 | 0.05 | 40.73 | `031a11ee` |
| xgboost | class_weight | 0.8231 | 0.6167 | 0.5489 | 0.1631 | 0.12 | 40.26 | `5269e25c` |

Tuned candidate (logreg/class_weight, run `0018d416`) on the **test** set:

| ROC-AUC | PR-AUC | precision | recall | F1 | Brier | threshold | cost/cust |
|---|---|---|---|---|---|---|---|
| 0.8541 | 0.6683 | 0.3596 | 0.9929 | 0.5280 | 0.1627 | 0.13 | 37.51 |

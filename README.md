# Feature Engineering Review — Bank Marketing

This prepared project is designed for a 10–15 minute Data Science desktop workflow.

## Public dataset
Source: UCI Machine Learning Repository — Bank Marketing.
The official dataset describes a classification problem where the target `y` indicates whether a client subscribed to a term deposit.

The recording workflow:
1. Review the GitHub repository/README.
2. Open `src/model_analysis.py` in VS Code.
3. Run the baseline model.
4. Inspect skewness and identify `campaign` as a right-skewed feature.
5. Apply `log1p(campaign)` as a feature-engineering change.
6. Run the updated model.
7. Open `results/model_comparison.csv` in Numbers.
8. Document the change in metrics and export it as `feature_engineering_review_deliverable.pdf`.

## Dataset setup
The UCI dataset should be downloaded before recording and placed at:
`data/bank.csv`

The setup script can download and prepare a small task subset.

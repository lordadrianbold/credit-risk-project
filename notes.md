# Project Notes

Keep a running log here as you build. This becomes the backbone of your README
and your answers in interviews — specifics beat "it worked."

## Week 1: Data understanding + cleaning

- Target definition (which loan_status values = default, which = paid, which excluded):
- Leakage columns excluded and why:
- Class balance (% default):
- Missingness patterns worth flagging:

## Week 2: Exploratory analysis + feature understanding

- Top 5-8 features that look predictive by eye, and your hypothesis why:
- Correlated feature pairs found:
- Missingness that turned out to be informative (not random):

## Week 3: Feature engineering + baseline model

- Engineered features added (e.g. debt-to-income, credit utilization):
- Logistic regression results (precision/recall/ROC-AUC):
- Strongest coefficients and what they mean:

## Week 4: Tree-based models + imbalance handling

- Random forest vs. XGBoost results:
- Imbalance handling approach used (class weighting vs. SMOTE) and which worked better:
- Final comparison table:

| Model | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Logistic regression | | | |
| Random forest | | | |
| XGBoost | | | |

## Week 5: Model interpretation + business framing

- SHAP findings — which features drive predictions most:
- Example of a manually walked-through prediction:
- Decision threshold chosen and why (business tradeoff reasoning):

## Week 6: Deployment + monitoring

- Where is it deployed (link)?
- What does the monitoring dashboard track?
- What would you improve with more time?



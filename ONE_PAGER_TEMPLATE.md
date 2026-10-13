# Explainable Credit Default Classification — Results (complete after running)

**Business question:** Can historic repayment patterns help flag likely next-month defaults for analyst review?

**Data:** UCI Default of Credit Card Clients (id 350). Number of valid records: [fill]; positive-class rate: [fill]. Split: [fill], random seed: [fill]. Exclusions/leakage review: [fill].

**Compared models:** Logistic Regression, Decision Tree, KNN, Random Forest, XGBoost. Tuning strategy: [fill].

**Held-out model comparison:** Best model [fill]; ROC-AUC [fill], PR-AUC [fill], recall [fill], precision [fill], F1 [fill], Gini [fill], KS [fill]. Selected threshold [fill] and business justification [fill].

**Global explanation:** Top features by permutation importance [fill]. SHAP summary and dependence indicate [fill].

**Local explanation:** Example row [fill], default probability [fill]. Leading SHAP contributions [fill]. LIME comparison: [agreement/disagreement + explanation].

**Performance vs interpretability:** Logistic Regression ROC-AUC [fill] versus XGBoost [fill]. Recommended model and why: [fill].

**Fairness and limitations:** Audit sex/age and possible proxies; report subgroup disparities [fill]. Observational historical data, no causal claims, no calibration verification, no deployment to real credit decisions. Future work: temporal validation, calibration, business-cost optimization, human review and fairness review.

**Recommended decision process:** Use scores to prioritize **manual** review and support case investigation, not automated rejection. Consult compliance experts before any real use.

*Attach measured plots/screenshots as space permits; export this template to a one-page PDF only after measured results are available.*

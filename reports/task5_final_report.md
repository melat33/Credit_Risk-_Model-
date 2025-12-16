# Bati Bank - Credit Risk Model Final Report

## Executive Summary

**Project**: Credit Risk Model for BNPL Service  
**Date**: 2025-12-16  
**Status**: COMPLETED SUCCESSFULLY  
**Best Model**: Logistic Regression  
**Performance**: ROC-AUC = 1.000

## Model Performance

| model                 |   test_roc_auc |   test_f1 |   test_precision |   test_recall |   false_negative_rate |   business_cost |
|:----------------------|---------------:|----------:|-----------------:|--------------:|----------------------:|----------------:|
| Logistic Regression   |       1        |  1        |         1        |      1        |            0          |               0 |
| Decision Tree         |       0.998155 |  0.995185 |         0.990415 |      1        |            0          |            3000 |
| Random Forest         |       1        |  0.998384 |         1        |      0.996774 |            0.00322581 |           10000 |
| XGBoost               |       1        |  1        |         1        |      1        |            0          |               0 |
| Random Forest (Tuned) |       1        |  0.998384 |         1        |      0.996774 |            0.00322581 |           10000 |

## Business Impact

* **Estimated Annual Savings**: $0
* **High-Risk Detection Rate**: 100.0%
* **False Positive Rate**: 0.0%
* **False Negative Rate**: 0.0%

## Basel II Compliance

| Requirement | Threshold | Actual | Status |
|------------|-----------|--------|--------|
| ROC-AUC | >= 0.7 | 1.000 | ✅ MET |
| False Negative Rate | <= 20% | 0.0% | ✅ MET |
| **Overall Compliance** | **Both requirements** | - | **✅ COMPLIANT** |

## Next Steps

1. **Deploy model to production API**
2. **Monitor model performance monthly**
3. **Retrain quarterly with new data**
4. **Regulatory reporting preparation**
5. **User training for risk analysts**

## Artifacts Generated

* 5 trained models with hyperparameter tuning
* MLflow experiment tracking with 6 runs
* Production model saved in `/models/best_model/`
* Complete documentation and unit tests in `/tests/`
* Business impact analysis in `/reports/`
* Feature importance analysis
* SHAP values for model interpretability

## Technical Specifications

**Model Type**: Logistic Regression  
**Framework**: Scikit-learn / XGBoost  
**Random Seed**: 42  
**Features Used**: 17  
**Training Samples**: 2,095  
**Validation Samples**: 524  
**Test Samples**: 1,123

---

*Report generated automatically by Bati Bank Credit Risk Modeling Team*  
*Confidential - For Internal Use Only*

def check_explainability(model):
    return hasattr(model, "coef_") or hasattr(model, "feature_importances_")

def calculate_pd(features, model):
    """Return predicted probability of default."""
    return model.predict_proba(features)[:, 1]

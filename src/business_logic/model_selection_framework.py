def select_model(models, metric_scores):
    return models[max(metric_scores, key=metric_scores.get)]

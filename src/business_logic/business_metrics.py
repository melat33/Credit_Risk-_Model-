def rfm_score(recency, frequency, monetary):
    return recency * -1 + frequency * 2 + monetary * 3

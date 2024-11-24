from feature_extraction import extract_features

def predict_name(context, model, vectorizer):
    """
    Predict the unredacted name based on the context.
    """
    features = extract_features(context)  # Extract features from the context
    X = vectorizer.transform([features['context']])  # Use only the context feature for now
    prediction = model.predict(X)
    return prediction[0]
from feature_extraction import process_data
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import nltk

# Ensure required resources are downloaded

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
def train_model(training_data):
    """
    Train the model using the training data.
    """
    # Extract features and labels
    features = process_data(training_data)
    labels = training_data["name"]
    
    # Vectorize features
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(features.to_dict(orient="records"))
    
    # Train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, labels)
    
    return model, vectorizer

def validate_model(model, vectorizer, validation_data):
    """
    Validate the model using the validation data and calculate performance metrics.
    """
    # Extract features and labels
    features = process_data(validation_data)
    X = vectorizer.transform(features.to_dict(orient="records"))
    y_true = validation_data['name']

    # Make predictions
    y_pred = model.predict(X)

    # Calculate precision, recall, and F1-score
    report = classification_report(y_true, y_pred, zero_division=1)
    print("Validation Performance Metrics:")
    print(report)
    
    with open("classification_report.txt", "w") as file:
        file.write(report)

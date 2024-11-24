import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk import pos_tag


# Data Loader
def load_data(file_path):
    """
    Load the unredactor data from a TSV file, skipping malformed lines.
    """
    try:
        headers = ["split", "name", "context"]
        data = pd.read_csv(file_path, sep="\t", names=headers, skiprows=0, on_bad_lines="skip")
        return data
    except FileNotFoundError:
        print(f"File not found at {file_path}. Ensure the path is correct.")
        return None


def split_data(data):
    """
    Split the data into training and validation sets.
    """
    training_data = data[data["split"] == "training"]
    validation_data = data[data["split"] == "validation"]
    return training_data, validation_data


# Feature Extraction
def extract_features(context, name=None):
    """
    Extract features such as previous and next words, POS tags, and word lengths.
    """
    tokenized_context = word_tokenize(context)
    pos_tags = pos_tag(tokenized_context)
    redaction_length = len(name) if name else 0
    context_length = len(context)
    word_count = len(tokenized_context)
    
    return {
        'context': context,  # Preserve original context for TF-IDF
        'redaction_length': redaction_length,  # Length of redaction
        'context_length': context_length,  # Length of the context
        'word_count': word_count,  # Total words in the context
        'pos_tags': " ".join(tag for _, tag in pos_tags),  # POS tags as a feature
        'redaction_ratio': redaction_length / context_length if context_length > 0 else 0,  # Redaction length to context length ratio
    }


def process_data(data):
    """
    Process the dataset to include extracted features.
    """
    feature_list = data.apply(
        lambda row: extract_features(row["context"], row["name"]), axis=1
    )
    features_df = pd.DataFrame(feature_list.tolist())
    return features_df


# Model Training and Validation
def train_model(training_data):
    """
    Train the model using the training data.
    """
    # Extract features
    features = process_data(training_data)
    labels = training_data['name']  # Target variable (names)

    # Use TfidfVectorizer for the context feature
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))  # Bi-grams and tri-grams
    X = vectorizer.fit_transform(features['context'])  # Use the 'context' column

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=None, random_state=42)
    model.fit(X, labels)

    return model, vectorizer


def validate_model(model, vectorizer, validation_data):
    """
    Validate the model using the validation data and calculate performance metrics.
    """
    # Extract features
    features = process_data(validation_data)
    labels = validation_data['name']

    # Transform validation context into TF-IDF features
    X_val = vectorizer.transform(features['context'])

    # Make predictions
    y_pred = model.predict(X_val)

    # Generate classification metrics
    report = classification_report(labels, y_pred, zero_division=1)
    print("Validation Performance Metrics:")
    print(report)
    return report


# Test Model
def predict_name(context, model, vectorizer, redaction_length=None):
    """
    Predict the unredacted name based on the context, prioritizing length matches.
    """
    X = vectorizer.transform([context])  # Directly transform the context using TF-IDF
    predictions = model.predict(X)

    # Filter predictions based on length
    if redaction_length:
        filtered_predictions = [
            pred for pred in predictions if len(pred) == redaction_length
        ]
        if filtered_predictions:
            return filtered_predictions[0]  # Return the first match
    return predictions[0]  # Fallback to the first prediction


def test_model(test_file, model, vectorizer, output_file="submission.tsv"):
    """
    Test the model on the test set and produce the submission file.
    :param test_file: Path to the test.tsv file
    :param model: Trained model
    :param vectorizer: Fitted vectorizer
    :param output_file: File to save the predictions
    """
    try:
        # Load the test data directly
        test_data = pd.read_csv(test_file, sep="\t", header=None, names=["id", "context"])
        
        # Ensure required columns exist
        if not set(["id", "context"]).issubset(test_data.columns):
            raise ValueError("Test file must have 'id' and 'context' columns.")
        
        # Convert IDs to integers
        test_data["id"] = test_data["id"].astype(int)

        # Make predictions
        predictions = []
        for _, row in test_data.iterrows():
            context = row["context"] if isinstance(row["context"], str) else str(row["context"])
            redaction_length = context.count("█")
            prediction = predict_name(context, model, vectorizer, redaction_length=redaction_length)
            predictions.append({"id": row["id"], "name": prediction})

        # Create the submission DataFrame
        submission_df = pd.DataFrame(predictions)

        # Save to submission.tsv
        submission_df.to_csv(output_file, sep="\t", index=False)
        print(f"Submission file saved to {output_file}.")

    except FileNotFoundError:
        print(f"Error: Test file {test_file} not found.")
    except Exception as e:
        print(f"Error during testing: {e}")


# Main Code
if __name__ == "__main__":
    # Load and split data
    file_path = "./data/unredactor.tsv"
    data = load_data(file_path)

    if data is not None:
        # Split into training and validation sets
        training_data, validation_data = split_data(data)

        if training_data is not None and validation_data is not None:
            # Train the model
            model, vectorizer = train_model(training_data)

            # Validate the model
            validate_model(model, vectorizer, validation_data)

            # Test the model and generate submission
            test_file = "./data/test.tsv"  # Path to the test file
            test_model(test_file, model, vectorizer)
        else:
            print("Error: Training or validation data is empty.")
    else:
        print("Error: Unable to load data.")

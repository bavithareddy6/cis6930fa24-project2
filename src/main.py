from data_loader import load_data, split_data
from model_training import train_model, validate_model
from test_model import test_model
import nltk

# Ensure NLTK resources are downloaded
nltk.data.path.append(r'C:\Users\bavit\AppData\Roaming\nltk_data')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')

# Load and split data
file_path = "./data/unredactor.tsv"
data = load_data(file_path)

if data is not None:
    # Split into training and validation sets
    training_data, validation_data = split_data(data)

    if training_data is not None and validation_data is not None:
        # Train the model
        model, vectorizer = train_model(training_data)

        # Validate the model and display performance metrics
        validate_model(model, vectorizer, validation_data)

        # Test the model on the test set
        test_file = "./data/test.tsv"  # Path to the test file
        test_model(test_file, model, vectorizer)
    else:
        print("Error: Training or validation data is empty.")
else:
    print("Error: Unable to load data.")

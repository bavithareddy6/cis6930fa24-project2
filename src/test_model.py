import pandas as pd
from predictor import predict_name
from data_loader import load_test_data
def test_model(test_file, model, vectorizer, output_file="submission.tsv"):
    """
    Test the model on the test set and produce the submission file.
    """
    try:
        # Load the test data
        test_data = load_test_data(test_file)

        if test_data is None or test_data.empty:
            print("Error: Test data could not be loaded or is empty.")
            return

        # Ensure required columns exist
        if not set(["id", "context"]).issubset(test_data.columns):
            raise ValueError("Test file must have 'id' and 'context' columns.")

        # Make predictions
        predictions = []
        for _, row in test_data.iterrows():
            context = row["context"] if isinstance(row["context"], str) else str(row["context"])
            prediction = predict_name(context, model, vectorizer)
            predictions.append((row["id"], prediction))

        # Create the submission DataFrame
        submission_df = pd.DataFrame(predictions, columns=["id", "name"])

        # Save to submission.tsv
        submission_df.to_csv(output_file, sep="\t", index=False)
        print(f"Submission file saved to {output_file}.")

    except FileNotFoundError:
        print(f"Error: Test file {test_file} not found.")
    except Exception as e:
        print(f"Error during testing: {e}")
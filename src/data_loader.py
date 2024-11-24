import pandas as pd

def load_data(file_path):
    """Load the unredactor data from a TSV file, skipping malformed lines."""
    try:
        headers = ['split', 'name', 'context']
        data = pd.read_csv(
            file_path, 
            sep="\t", 
            names=headers, 
            skiprows=1, 
            on_bad_lines='skip'  # Skip problematic lines
        )
        return data
    except FileNotFoundError:
        print(f"File not found at {file_path}. Ensure the path is correct.")
        return None

def split_data(data):
    """Split the data into training, validation, and testing."""
    try:
        training_data = data[data['split'] == 'training']
        validation_data = data[data['split'] == 'validation']
        
        return training_data, validation_data
    except KeyError as e:
        print(f"KeyError: {e}. Verify the column names in your dataset.")
        return None, None, None

def load_test_data(file_path):
    """
    Load the test data from a TSV file and ensure the columns are correctly named.
    """
    try:
        # Read the test data
        data = pd.read_csv(file_path, sep="\t", names=["id", "context"], skiprows=0)
        return data
    except FileNotFoundError:
        print(f"File not found at {file_path}. Ensure the path is correct.")
        return None
    except Exception as e:
        print(f"Error while loading test data: {e}")
        return None


if __name__ == "__main__":
    file_path = "./data/unredactor.tsv"  # Path to the dataset
    data = load_data(file_path)
    if data is not None:
        print(data.columns)  # Verify column names
        training_data, validation_data = split_data(data)
        if training_data is not None:
            print(f"Training Samples: {len(training_data)}")
            print(f"Validation Samples: {len(validation_data)}")
            #print(f"Testing Samples: {len(testing_data)}")

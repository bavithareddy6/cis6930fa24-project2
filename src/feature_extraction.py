import re
import pandas as pd
from nltk import pos_tag, word_tokenize

def extract_features(context, name=None):
    """
    Extract features such as previous and next words, POS tags, and redaction length.
    """
    # Find the redaction marker
    redaction_index = context.find("█")
    words_before = re.findall(r'\w+', context[:redaction_index])
    words_after = re.findall(r'\w+', context[redaction_index:])

    # Tokenize the context and extract POS tags
    tokenized_context = word_tokenize(context)
    pos_tags = pos_tag(tokenized_context)

    return {
        'previous_word': words_before[-1] if words_before else '',
        'next_word': words_after[0] if words_after else '',
        'redaction_length': len(name) if name else 0,
        'pos_tags': " ".join([tag for _, tag in pos_tags])  # Aggregate POS tags
    }

def process_data(data):
    """
    Process the dataset to include extracted features.
    """
    feature_list = data.apply(
        lambda row: extract_features(row["context"], row["name"]), axis=1
    )
    features_df = pd.DataFrame(feature_list.tolist())
    features_df['context'] = data['context']
    return features_df

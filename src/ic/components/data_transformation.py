import pandas as pd
from transformers import AutoTokenizer

# Load the dataset
input_path = '/Users/suyash/Desktop/Intent-classification-/artifacts/data_ingestion/train/train.csv'
df = pd.read_csv(input_path)

# Display the first few rows to understand the structure
print("Loaded data:")
print(df.head())

# Use correct column names based on the provided data
text_column = 'utterance'  # Column containing text data
label_column = 'intent'    # Column containing label data

# Initialize tokenizer (using DeBERTa for this example)
tokenizer_name = "microsoft/deberta-base"  # Choose the appropriate model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Tokenize the text data
def tokenize_data(text):
    return tokenizer(
        text,
        padding='max_length',  # Pad to max length of the model
        truncation=True,       # Truncate sequences longer than the model's max length
        max_length=128,        # Adjust according to model input size
        return_tensors="pt"    # PyTorch tensors
    )

# Apply tokenization to the text column
df['tokens'] = df[text_column].apply(lambda x: tokenize_data(x).input_ids[0].tolist())

# Display tokenized data
print("\nSample tokenized data:")
print(df[['utterance', 'tokens', label_column]].head())

# Save the tokenized data to a new CSV file
output_path = '/Users/suyash/Desktop/Intent-classification-/artifacts/data_ingestion/train/tokenised.csv'
df.to_csv(output_path, index=False)

print(f"\nTokenized data saved to {output_path}")

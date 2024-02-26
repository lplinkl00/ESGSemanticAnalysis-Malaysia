from gensim.models import Word2Vec
import json
import pandas as pd

def load_json_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

json_files = ['PDF2JSON\i-ESG Starter Kit_cleaned.json', 'PDF2JSON\SEDG-Full-Guide_cleaned.json', 'PDF2JSON\Bursa Sustainability Reporting Guide_cleaned.json']  # List of your JSON file paths

all_texts = []
for file_path in json_files:
    preprocessed_texts = load_json_data(file_path)
    all_texts.extend(preprocessed_texts)

file_name = 'PDF2JSON\SEDG-Full-Guide_cleaned.json'
preprocessed_data = load_json_data(file_name)
model = Word2Vec(sentences=preprocessed_data, vector_size=100, window=5, min_count=1, workers=4)  # Fit the model
#Save model
model.save("PDF2JSON\sedg.model")

# Find similar words
# similar_words = model.wv.most_similar('report', topn=10)

# Get the vector for a word
# word_vector = model.wv['environment']

# df = pd.DataFrame(similar_words, columns=['Word', 'Similarity'])

# Save the DataFrame to an Excel file
# file_parts = file_name.split('.')
# new_file_path = file_parts[0] + '_Similar_Words.' + 'xlsx'
# df.to_excel(new_file_path, index=False)


# Calculating distances

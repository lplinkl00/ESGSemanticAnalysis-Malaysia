import pandas
import numpy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
import string
import json
import re
from langdetect import detect


def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

# Load JSON data from a file
def load_json_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

def remove_undefined_characters(text):
    # This pattern matches any character that is not a letter, number, or common punctuation
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s,.!?-]', '', text)
    return cleaned_text


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    #remove undefined characters
    text = remove_undefined_characters(text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize text
    tokens = wordpunct_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Filter out non-English words using langdetect
    english_tokens = [word for word in tokens if is_english(word)]
    # Stemming (optional)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in english_tokens]
    return tokens

# Example preprocessing of the first page's content (modify according to your JSON structure)
def process_all_pages(json_data):
    preprocessed_pages = []
    for page in json_data['pages']:
        preprocessed_text = preprocess_text(page['content'])
        preprocessed_pages.append(preprocessed_text)
    return preprocessed_pages

file_name = "PDF2JSON\Bursa Sustainability Reporting Guide.json"
json_data = load_json_data(file_name)
preprocessed_data = process_all_pages(json_data)
# Assuming `preprocessed_data` is your preprocessed content
# and 'your_data_file.json' is the original file name

def write_preprocessed_json(preprocessed_data, original_file_path):
    # Split the original file path to insert "_cleaned" before the .json extension
    file_parts = original_file_path.split('.')
    cleaned_file_path = file_parts[0] + '_cleaned.' + file_parts[1]
    
    # Write the preprocessed data to the new file
    with open(cleaned_file_path, 'w', encoding='utf-8') as cleaned_file:
        json.dump(preprocessed_data, cleaned_file, ensure_ascii=False, indent=4)

# Example usage
write_preprocessed_json(preprocessed_data, file_name)




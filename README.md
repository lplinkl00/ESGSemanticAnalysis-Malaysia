# Semantic Analysis of ESG Standards

# Background

Comparing three ESG guidelines in Malaysia:

1. Simplified ESG Disclosure Guide 2023 (SEDG)
2. i-ESG Starter Kit 2023
3. Bursa Sustainability Reporting Guide 2022

To see how much they have in common at a data level and throwing the discussion out to public on LinkedIn to both see if there is any feedback to both the idea and my methodology. 

[https://ai-jobs.net/insights/semantic-analysis-explained/#what-is-semantic-analysis](https://ai-jobs.net/insights/semantic-analysis-explained/#what-is-semantic-analysis)

Code writing assisted by ChatGPT 4 Feb 2024 release. No Plugins used. Python 3.10. Libraries will be listed in the steps taken below.

# Process

### Convert files from PDF to JSON

Started by converting all each of the guidelines from pdf to json files with PymuPDF

PymuPDF = 1.23.25

json

### Preprocess and Clean JSON

The JSON files directly converted from PDF carry lots of images and redundant text such as page numbers that distort the data as the model in the future may create false associations. 

Libraries used: 

NLTK = 3.8.1

langdetect = 1.0.9

json

re

The data is cleaned with the following steps:

1. lowercase all text
2. remove punctuation 
3. remove undefined text
4. remove numbers 
5. tokenize
6. remove stopwords
7. remove non-english words - to remove alternate text of images
8. Stem tokens

### Create Model

Gensim = 4.3.2

After the data is preprocessed there are 4 different models to create. One for each document and then all documents combined.

Word2Vec was chosen to see which words were most strongly related. Assuming that each document is written differently, the models should be totally different in their most common words and relations. 

## Visualization

To visualize the models together in one graph, without combining the models together, so that the models can be seen in relation to one another.

In order to plot the models onto a graph a dimensionality reduction algorithm needs to be used to, in an overly simplified explaination, convert the words stored in the models to x,y coordinates. 

In this project UMAP, t-SNE and PCA were tested to see how the models would look. PCA is used for the final graph due to its consistent output over multiple runs.

To achieve this 50 common words were extracted from the three models to set a benchmark without using too much time for each run. The list of words are then used to extract the corresponding vectors from each model. After PCA is applied to the vectors, the reduced vectors gathered are split into seperate lists following the order of the models being combined. Such that if model 1 + model 2 + model 3 is the order of the combination of models to get the common words, the vectors are split by model 1, 2 and 3 in that order. This is important to ensure that the labels when plotting the vectors are correct.

Finally a scatter plot is used to display the vectors.






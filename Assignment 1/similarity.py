# -------------------------------------------------------------------------
# AUTHOR: Joseline Ly
# FILENAME: similarity.py
# SPECIFICATION: This program reads a CSV file containing documents, 
#   constructs a document-term matrix using binary encoding, computes
#   pairwise cosine similarities between the documents, and identifies
#   the two most similar documents based on these similarities.
# FOR: CS 4440 (Data Mining) - Assignment #1
# TIME SPENT: 5.5 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy or pandas.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
    if i > 0: #skipping the header
      documents.append(row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection using the white space as your character delimiter.
#--> add your Python code here

docTermMatrix = []
allWords = set()
binaryEncoding = []

for wordRow in documents:
  # Separate all the different words into list words
  words = wordRow[1].split()
  # Remove any numbers from the list words
  filteredWords = [word for word in words if not word.isdigit()]
  # Create a set of unique words
  allWords.update(filteredWords)

for wordRow in documents:
    words = wordRow[1].split()
    filteredWords = [word for word in words if not word.isdigit()]
    
    # Create a set for the current document's words for fast lookup
    docWordSet = set(filteredWords)
    
    # Create the binary encoding row
    binary_encoding_row = []
    for word in allWords:
        if word in docWordSet:
            binary_encoding_row.append(1)
        else:
            binary_encoding_row.append(0)      
    docTermMatrix.append(binary_encoding_row)

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here

mostSimilarDocs = (0, 0)
highestSimilarity = -1  # Initialize to -1 to ensure any similarity will be higher

# For debugging purposes, I am storing all similarities in a list. This will allow
#   me to verify that the number of similarities total is 80,200 or C(401, 2).
# allSimilarities = []

for i in range(len(docTermMatrix)):
  for j in range(i + 1, len(docTermMatrix)):
    similarityValue = cosine_similarity([docTermMatrix[i]], [docTermMatrix[j]])[0][0]
    
    # Appending all similarities to the list for debugging purposes
    # allSimilarities.append(similarityValue)

    if similarityValue > highestSimilarity:
      highestSimilarity = similarityValue
      # Adding 1 to i and j to match document numbering (1-indexed)
      mostSimilarDocs = (i + 1, j + 1)

# Print length of allSimilarities for debugging purposes
# print(f"Length of allSimilarities: {len(allSimilarities)}")

# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
print(f"The most similar documents are document {mostSimilarDocs[0]} and document {mostSimilarDocs[1]} with cosine similarity = {highestSimilarity}")
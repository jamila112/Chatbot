import random
import json

import torch

from model import NeuralNet
#from nltk_utils import bag_of_words, tokenize
#####################################################################################################

import nltk
import re
import nltk
import numpy as np
from gensim import corpora, similarities
from gensim.models import LsiModel
from nltk.corpus import stopwords
#nltk.download("stopwords")
#nltk.download("wordnet")
from numpy import zeros
from scipy.linalg import svd
import re
import numpy as np
import pandas as pd
import nltk
from nltk.stem import wordnet
from nltk import pos_tag
from scipy import linalg, spatial
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer, TfidfVectorizer)
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import pairwise_distances
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
df = pd.read_excel(r'/home/jamila/PycharmProjects/ri/Classeur (1).xlsx')
#df = pd.read_excel(r'/home/jamila/PycharmProjects/ri/chatbot.xlsx',sheet_name='chatbot.xlsx')
#print(df)
import nltk
nltk.download('omw-1.4')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
##################################################################################


#####################################################################################

def text_normalization(text):
  text=str(text).lower()
  spl_char_text=re.sub(r'[^ a-z]','',text)
  tokens=nltk.word_tokenize(spl_char_text)
  lema=wordnet.WordNetLemmatizer()
  tags_list=pos_tag(tokens,tagset=None)
  lema_words=[]
  for token,pos_token in tags_list:
    if pos_token.startswith('V'):
      pos_val='v'
    elif pos_token.startswith('J'):
      pos_val='a'
    elif pos_token.startswith('R'):
      pos_val='r'
    else:
      pos_val='n'
    lema_token=lema.lemmatize(token,pos_val)
    lema_words.append(lema_token)
  return " ".join(lema_words)
df['lemmatized_text']=df['Context'].apply(text_normalization)
#print(df)

import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')
import re
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
 
NUM_TOPICS = 10
STOPWORDS = stopwords.words('english')
 
def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text
 
# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = []
for text in df['Context']:
    tokenized_data.append(clean_text(text))
#print(tokenized_data)



#####################################################################################################


bot_name = "Bot"

def get_response(msg):

                query_norm = text_normalization(msg)
                print(query_norm)
                query_clean = clean_text(msg)
                print(query_clean)

                # Build a Dictionary - association word to numeric id
                dictionary = corpora.Dictionary(tokenized_data)
                print("\n")
                print(dictionary)

                # Transform the collection of texts to a numerical form
                corpus = [dictionary.doc2bow(text) for text in tokenized_data]
                print(corpus)
                print("\n")




                # Build the LSI model
                lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
                print(lsi_model)


                #Derivation of Term Document Matrix of Training Document Word Stems = M' x [Derivation of T]
                print("LSI Vectors of Training Document Word Stems: ",
                      [lsi_model[tokenized_data] for tokenized_data in corpus])





                #calculate cosine similarity matrix for all training document LSI vectors
                cosine_similarity_matrix = similarities.MatrixSimilarity(lsi_model[corpus])
                print("Cosine Similarities of LSI Vectors of Training Documents:",
                    [row for row in cosine_similarity_matrix])



                #calculate LSI vector from word stem counts of the test document and the LSI model content
                vector_lsi_test = lsi_model[dictionary.doc2bow(query_clean)]
                #print("LSI Vector Test Document:", vector_lsi_test)




                #perform a similarity query against the corpus
                cosine_similarities_test = cosine_similarity_matrix[vector_lsi_test]
                #print("Cosine Similarities of Test Document LSI Vectors to Training Documents LSI Vectors:",cosine_similarities_test)
                      



                #get text of test documents most similar training document
                most_similar_document_test = df['Response'][np.argmax(cosine_similarities_test)]
                #print("Most similar Training Document to Test Document:", most_similar_document_test)
                

                return most_similar_document_test
              
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)


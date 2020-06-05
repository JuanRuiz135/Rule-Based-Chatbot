# Libraries
import string # to process standard python strings 
import pandas as pd
import numpy as np

# For preprocessing of the data
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
# Generating response with document similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#  Initialise AIML agent
import aiml
# Initialise Wikipedia agent
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)

# Initialize the Kernel (the AIML file). 
kern.bootstrap(learnFiles="mybot-basic.xml")

# Reading in the data
df = pd.read_csv('questionanswer-dogs.txt', sep=':')

# Pre-procecssing the raw text
stopwords_list = stopwords.words('english') # set stopwords to english

lemmatizer = WordNetLemmatizer() # Returns the input word unchanged if it cannot be found in WordNet

def my_tokenizer(doc):
    words = word_tokenize(doc) # tokenize the document
    
    pos_tags = pos_tag(words) # Parts of speech tagging
    
    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list] # detects stopwords
    
    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation] # detects punctuation
    
    lemmas = [] # function for finding lemmas in the wordnet
    for w in non_punctuation:
        if w[1].startswith('J'):
            pos = wordnet.ADJ
        elif w[1].startswith('V'):
            pos = wordnet.VERB
        elif w[1].startswith('N'):
            pos = wordnet.NOUN
        elif w[1].startswith('R'):
            pos = wordnet.ADV
        else:
            pos = wordnet.NOUN    
            
        lemmas.append(lemmatizer.lemmatize(w[0], pos))
        
    return lemmas
   
# Define function response which search for utterance of keywords in inputs and returns possible answer.
def response(user_response):
    doggy_answer=''
     # Convert the file to a matrix of TF-IDF features.
    TfidfVec = TfidfVectorizer(tokenizer=my_tokenizer)
    tfidf = TfidfVec.fit_transform(tuple(df['Question']))# return term-document matrix. 
    
    query_vector = TfidfVec.transform([user_response]) # matrix of the user input
    similarity = cosine_similarity(query_vector, tfidf) # computes similarity as the normalized dot product of X and Y
    max_simil = np.argmax(similarity, axis=None) # finds max similarity in the document
    doggy_answer = doggy_answer+df.iloc[max_simil]['Answer'] # answer with major similarity gets chosen
    return doggy_answer

# Welcome user

print("Welcome! My name is Doggy and I know all about how to take care of dogs.",
      "Please feel free to ask questions about anything dog related.")

# Main loop

while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
            wpage = wiki_wiki.page(params[1])
            if wpage.exists():
                print(wpage.summary[0:300] + '...')
                print("Learn more at", wpage.canonicalurl)
            else:
                print("Sorry, I don't know what that is.")
        elif cmd == 2:
           userInput=userInput.lower()
           print(response(userInput)) 
        elif cmd == 99:
            print("I did not get that, please try again.")
    else:
        print(answer)



    
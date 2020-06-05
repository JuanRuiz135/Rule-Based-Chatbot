"""
This code is part of a coursework in development for the 
Artificial Intelligence module at Nottingham Trent University

"""
# Imports
import string # to process standard python strings 
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# For preprocessing of the data
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag

# Generating response with document similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# GUI
import os
from flask import Flask, render_template, request,flash
from werkzeug.utils import secure_filename
from chatterbot import ChatBot

# Predictive model
from skimage.io import imread
from PIL import Image
from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split
from keras.utils import np_utils, Sequence
from keras.applications.densenet import preprocess_input

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


#####################################################################################
# Compile the model for image classification
from keras.models import load_model
# load model
model = load_model('weights.h5')
print('Image Classification model loaded')
# Loading the data labels for the dogs
breeds = os.listdir("dogs/images/Images/")

num_breeds = len(breeds)

# Mapping
label_maps = {}
label_maps_rev = {}
for i, v in enumerate(breeds):
    label_maps.update({v: i})
    label_maps_rev.update({i : v})

# Function to predict the dog breed and return the name
def predict_breed(filename):
    # open image, resize it
    img = Image.open(filename)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img.save(filename)
    # predict
    img = imread(filename)
    img = preprocess_input(img)
    probs = model.predict(np.expand_dims(img, axis=0))
    for idx in probs.argsort()[0][::-1][:1]:
        breed_pred = label_maps_rev[idx].split("-")[-1]
    return breed_pred

# Define function to see if the file being uploaded is an image
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#######################################################################################


################################# Part of last submission ############################

# Import the function that compile and predict from the transformer network
from predict_net import compile_load_model as transformer_net_compile
from predict_net import predict as transformer_net_predict
# compile the model for bot responses
transformer_model =  transformer_net_compile()
# Import the function that returns string with best foods from the genetic algorithm
from Genetic_Algorithm import best_foods as dog_food_calc 

######################################################################################


#######################################################################################
# TF/IDF feature
# Reading in the data for the question answer pair file
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
   
# Define function response which search for utterance of keywords in inputs 
# and returns possible answer.
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
###############################################################################################################

           
#######################################################
# Initializer first order logic based agent
from simple_sem import sheltersKnowledge as first_order_logic
import nltk
v = """
Boran => br
Berneses => {}
Huskies => {}
Sheperds => {}
Retrievers => {}
Pomskies => {}
Bernese => {}
Husky => {}
Sheperd => {}
Retriever => {}
Pomsky => {}
Barkers => dr1
Doggoland => dr2
Puppies => dr3
be_in => {}
are_in => {}
is_in => {}
is_named => {}
"""
folval = nltk.Valuation.fromstring(v)
grammar_file = 'simple-sem.fcfg'

objCounter = 0
first_order_logic()
######################################################


################### Main loop #######################
nameCounter = 0
ans = ""
##################### App GUI #####################################################
# Files upload folder for the app
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create a flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
bot = ChatBot('Norman')
###################################################################################

# Render website
@app.route("/", methods=['GET', 'POST'])
def home():
    print('template rendered')    
    return render_template("home.html")


# File upload function       
@app.route("/upload_file", methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return print('Please select file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))      
    return filename
 
# Function to get bots response 
@app.route("/get")
def get_bot_response():
    while True:
        #get user input
        try:
            userInput = request.args.get('msg')
            userInput = str(userInput)
        except (KeyboardInterrupt, EOFError) as e:
            ans = "Bye!"
            return ans
            break
        #pre-process user input and determine response agent (if needed)
        responseAgent = 'query'
        #activate selected response agent
        if responseAgent == 'query':
            answer = kern.respond(userInput)
        #post-process the answer for commands
        if answer[0] == '#':
            params = answer[1:].split('$')
            cmd = int(params[0])
            if cmd == 0:
                ans = params[1]
                return ans 
                break
            elif cmd == 1:
                wpage = wiki_wiki.page(params[1])
                if wpage.exists():
                    ans1 = wpage.summary[0:300] + '...' + 'Learn more at ' + wpage.canonicalurl
                    return ans1
                else:
                    ans = "Sorry, I don't know what that is."
                    return ans
            elif cmd == 2:
                userInput=userInput.lower()
                return response(userInput)
            ######################################################
            # CFG phrases #
            ######################################################
            elif cmd == 4: # I will buy x from y
                o = 'o' + str(objCounter)
                folval['o' + o] = o #insert constant
                if len(folval[params[1]]) == 1: #clean up if necessary
                    if ('',) in folval[params[1]]:
                        folval[params[1]].clear()
                folval[params[1]].add((o,)) #insert dog
                folval["be_in"].add((o, folval[params[2]])) #insert location
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                sent = 'some ' + params[1] + ' are_in ' + params[2]
                results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                if results[2] == True:
                    ans = "There's some " + params[1] + " in " + params[2] + "."
                    return ans
                else:
                    ans = "No."
                    return ans
            elif cmd == 5: #Are there any x in y
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                sent = 'some ' + params[1] + ' are_in ' + params[2]
                results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                if results[2] == True:
                    ans = "Yes. There are some " + params[1] + " in " + params[2] + "."
                    return ans
                else:
                    ans = "No."
                    return ans
            elif cmd == 7: # Which dogs are in ...
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                e = nltk.Expression.fromstring("be_in(x," + params[1] + ")")
                sat = m.satisfiers(e, "x", g)
                counter = 0
                ans = "The list of dogs available are: "
                if len(sat) == 0:
                     ans = "None."
                     return ans
                else:
                    #find satisfying objects in the valuation dictionary,
                     #and print their type names
                     sol = folval.values()
                     for so in sat:
                         for k, v in folval.items():
                             if counter == 5:
                                 return ans
                                 break
                             if len(v) > 0:
                                 vl = list(v)
                                 if len(vl[0]) == 1:
                                     for i in vl:
                                         if i[0] == so:
                                             if counter == 4:
                                                 ans = ans + k + ". "
                                                 counter = counter + 1
                                             else:
                                                 ans = ans + k + ", "
                                                 counter = counter + 1
                                                 break
                                                
                                            
            elif cmd == 8: # I will buy a x from y
                o = 'o' + str(objCounter)
                folval['o' + o] = o #insert constant
                if len(folval[params[1]]) == 1: #clean up if necessary
                    if ('',) in folval[params[1]]:
                        folval[params[1]].clear()
                folval[params[1]].add((o,)) #insert dog
                folval["be_in"].add((o, folval[params[2]])) #insert location
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                sent = 'some ' + params[1] + ' is_in ' + params[2]
                results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                if results[2] == True:
                    ans = "Yes. You can buy your " + params[1] + " from " + params[2]
                    return ans
                else:
                    ans = "No."
                    return ans
            elif cmd == 9: #Did i buy my x from y
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                sent = 'some ' + params[1] + ' is_in ' + params[2]
                results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                if results[2] == True:
                    ans = "Yes. You said you bought the " + params[1] + " from " + params[2]
                    return ans
                else:
                    ans = "No."
                    return ans
            elif cmd == 10: #Did i buy x from y
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                sent = 'some ' + params[1] + ' are_in ' + params[2]
                results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                if results[2] == True:
                    ans = "Yes. You bought your " + params[1] + " from " + params[2]
                    return ans
                else:
                    ans = "No."
                    return ans
                
            elif cmd == 12: # my x is named y
                o = 'o' + str(objCounter)
                if len(folval[params[1]]) == 1: #clean up if necessary
                    if ('',) in folval[params[1]]:
                        folval[params[1]].clear()
                #insert name of breed
                folval[params[1]].add((o,))
                if len(folval["is_named"]) == 1: #clean up if necessary
                    if ('',) in folval["is_named"]:
                        folval["is_named"].clear()
                folval["is_named"].add((o, folval[params[2]])) # name dog
                ans = params[2] + " seems like a nice dog."
                return ans
            
            ######################################################
            # Final submission implementation #
            ######################################################
            elif cmd == 19: 
                # if they ask for which is the best food make the calc and send the response
                ans = dog_food_calc()
                return ans
            elif cmd == 99: 
                # for everything else, send it to be predicted by the transformer network
                userInput=userInput.lower()
                return transformer_net_predict(userInput)
                
            
        else:
            return answer
        
        
# Makes the classification of the image and returns information about the dog breed.     
@app.route("/getClassification")
def get_image_classification():
    while True:
        #get user input
        try:
            userInput = request.args.get('msg')
            userInput = str(userInput)
        except (KeyboardInterrupt, EOFError) as e:
            ans = "Bye!"
            return ans
            break
        #pre-process user input and determine response agent (if needed)
        responseAgent = 'query'
        #activate selected response agent
        if responseAgent == 'query':
            dog = predict_breed(userInput)
            wpage = wiki_wiki.page(dog)
            if wpage.exists():
                ans1 = 'The image is of a ' + dog + '. ' + wpage.summary[0:400] + '...' + 'Learn more at ' + wpage.canonicalurl
                return ans1
            else:
                ans = "Sorry, I don't have information about that dog."
                return ans
        

    
# Start the server, continuously listen to requests.
if __name__ == "__main__":    
    app.run(threaded=False)
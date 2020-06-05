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

from keras.models import load_model
# load model
model = load_model('weights.h5')
print('Model loaded')
# summarize model.
# model.summary()

#--------------------------------------------------------------
# Loading the data and preprocessing
breeds = os.listdir("dogs/images/Images/")

num_breeds = len(breeds)

# Mapping
label_maps = {}
label_maps_rev = {}
for i, v in enumerate(breeds):
    label_maps.update({v: i})
    label_maps_rev.update({i : v})
print("Mapping done")
    
def label_target_path():
    paths = list()
    labels = list()
    targets = list()
    for breed in breeds:
        base_name = "./data/{}/".format(breed)
        for img_name in os.listdir(base_name):
            paths.append(base_name + img_name)
            labels.append(breed)
            targets.append(label_maps[breed])
    return paths, labels, targets

paths, labels, targets = label_target_path()

assert len(paths) == len(labels)
assert len(paths) == len(targets)

targets = np_utils.to_categorical(targets, num_classes=num_breeds)

#--------------------------------------------------------------------
# copy from https://www.kaggle.com/gabrielloye/dogs-inception-pytorch-implementation
# augments quality of the image by reducing background noise

batch_size = 64


class ImageGenerator(Sequence):
    
    def __init__(self, paths, targets, batch_size, shape, augment=False):
        self.paths = paths
        self.targets = targets
        self.batch_size = batch_size
        self.shape = shape
        self.augment = augment
        
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_paths = self.paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        x = np.zeros((len(batch_paths), self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)
        y = np.zeros((self.batch_size, num_breeds, 1))
        for i, path in enumerate(batch_paths):
            x[i] = self.__load_image(path)
        y = self.targets[idx * self.batch_size : (idx + 1) * self.batch_size]
        return x, y
    
    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
        image = imread(path)
        image = preprocess_input(image)
        if self.augment:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.CropAndPad(percent=(-0.25, 0.25)),
                    iaa.Crop(percent=(0, 0.1)),
                    iaa.Sometimes(0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])
            ], random_order=True)
            image = seq.augment_image(image)
        return image
#---------------------------------------------------------------------------

# Train test splits       
X_train, X_test, y_train, y_test = train_test_split(paths, 
                                                  targets,
                                                  test_size=0.20, 
                                                  random_state=19)

train_generator = ImageGenerator(X_train, y_train, batch_size=32, shape=(224,224,3), augment=True)
test_generator = ImageGenerator(X_test, y_test, batch_size=32, shape=(224,224,3), augment=False)
print("Train splits done")

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
 
#-------------------------------------------------------------

# Files upload folder for the app
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create a flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
bot = ChatBot('Norman')

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

# Define function to see if the file being uploaded is an image
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
# Main loop

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
            elif cmd == 99:
                ans = "I did not get that, please try again."
                return ans
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
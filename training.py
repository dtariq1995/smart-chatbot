import random
import json
import pickle
from tabnanny import verbose
import numpy as np
import nltk
import tensorflow as tf
from tensorflow import keras
from nltk.stem import WordNetLemmatizer
#from tensorflow.keras import Layers
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())   # load data

words = []
classes = []
documents = []
skip_characters = ['?','!', ',', '.']

for intent in intents['intents']:   # access the intents key from the intents.json file
    for pattern in intent['patterns']:   # read patterns for what users may say
        word_list = nltk.word_tokenize(pattern)   # tokenize all the words from patterns 
        words.extend(word_list)   # add tokenized words from pattern to the words list
        documents.append((word_list, intent['tag']))   # assign tag on a per word list basis so we know which tag goes to which word list
        if intent['tag'] not in classes:   # if the tag hasn't been added to the classes list yet, it will be added
            classes.append(intent['tag'])
            

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in skip_characters]   # lemmatize the words and add them to words list is not in the skip characters list
words = sorted(set(words))   # eliminate word duplicates and sort words

classes = sorted(set(classes))   # elimate duplicate classes just in case, and sort classes

pickle.dump(words, open('words.pkl', 'wb'))   # save words to file
pickle.dump(classes, open('classes.pkl', 'wb'))   # save classes to file

training = [] 
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0] 
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]   # lemmatize and lowercase the words in word_patterns
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)   # if this word is in word patterns, then add a 1 to bag, else add a 0 to bag
        
    output_row = list(output_empty)   # copying list
    output_row[classes.index(document[1])] = 1   # set this index in the output row to 1
    training.append([bag, output_row])   # append to training list, after this all document data will be in training list
    
random.shuffle(training)   # shuffle data
training = np.array(training)   # turn into numpy array

train_x = list(training[:, 0])   # features used to train neural network
train_y = list(training[:, 1])   # labels used to train neural network

model = Sequential()   # building a simple sequential model
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))   # first layer is dense layer, input shape is based on lenght of training data
model.add(Dropout(0.5))   # dropout to avoid overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))   # softmax scales the accuracy scores

sgd = SGD(lr=0.01, decay=1e-6, momentum=0/9, nesterov=True)   # adding an sgd optimizer
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])   # compile model, interested in accuracy 

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)  
model.save('chatbot_model.h5', hist)   # save model
print("Done")

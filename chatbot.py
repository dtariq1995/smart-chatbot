import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

bot_name = "Botty the Bot"   # give bot a cool name

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())   # load data

words = pickle.load(open('words.pkl', 'rb'))   # load words
classes = pickle.load(open('classes.pkl', 'rb'))   # load classes
model = load_model('chatbot_model.h5')   # load model

def clean_up_sentence(sentence):   
    sentence_words = nltk.word_tokenize(sentence)   # tokenize sentence
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]   # lemmatize words
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)   # tokenize and lemmatize sentences
    bag = [0] * len(words)   # create initial bag full of 0s
    for w in sentence_words:   # if word is in words file list, then add 1 to bag
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):   
    bow = bag_of_words(sentence)   # get bag of words of sentence
    res = model.predict(np.array([bow]))[0]   # pass numpy array of bag of words list to the model
    ERROR_THRESHOLD = 0.25   
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]   # ignore words that are over the error threshold
    
    results.sort(key=lambda x: x[1], reverse=True)   # use anonymous function to sort in descending order
    return_list= []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})   # return class and probability of it being that class
        
    return return_list

def get_response(intents_list, intents_json):   # get response from bot
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:   # bot will choose random response from the most likely class
            result = random.choice(i['responses'])
            break
    return result

print("Launching GUI")

def main():   # main function calling all functions for bots operation

    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    
    return res
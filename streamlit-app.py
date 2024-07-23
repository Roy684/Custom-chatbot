import streamlit as st
import random
import json
import torch
import nltk
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import numpy as np

nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the intents and model
with open('new_intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"

# Function to get a response from the model
def get_response(message):
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = np.array(X).reshape(1, -1)
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                return response
    else:
        return "I do not understand..."

# Streamlit app
st.title("Chatbot")

if 'responses' not in st.session_state:
    st.session_state.responses = []

user_input = st.text_input("You: ", key="input")

if st.button("Send"):
    if user_input:
        response = get_response(user_input)
        st.session_state.responses.append((user_input, response))

if st.session_state.responses:
    for user_message, bot_response in st.session_state.responses:
        st.write(f"You: {user_message}")
        st.write(f"{bot_name}: {bot_response}")

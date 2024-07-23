from flask import Flask, render_template, request, jsonify
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('new_intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json["message"]
    print(f"Received message: {message}")  # Debug print
    sentence = tokenize(message)
    print(f"Tokenized message: {sentence}")  # Debug print
    X = bag_of_words(sentence, all_words)
    print(f"Bag of words: {X}")  # Debug print
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    print(f"Model output: {output}")  # Debug print
    _, predicted = torch.max(output, dim=1)
    print(f"Predicted tag index: {predicted.item()}")  # Debug print

    tag = tags[predicted.item()]
    print(f"Predicted tag: {tag}")  # Debug print

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(f"Prediction probability: {prob.item()}")  # Debug print

    if prob.item() > 0.70:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"Response: {response}")  # Debug print
                return jsonify({"response": response})
    else:
        print("Response: I do not understand...")  # Debug print
        return jsonify({"response": "I do not understand..."})
    
if __name__ == "__main__":
    app.run(debug=True)

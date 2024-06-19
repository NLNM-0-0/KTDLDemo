import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizerFast, BertForSequenceClassification

model_dir = "models/AGNews/"

tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    return predicted_class

app = Flask(__name__)
CORS(app)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json(force=True)
    text = data['text']
    predicted_class = classify_text(text)
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
from transformers import DebertaV2Tokenizer, TFDebertaV2ForSequenceClassification
import tensorflow as tf

app = Flask(__name__)

model_id="microsoft/deberta-base"
MODEL_PATH = "/Users/suyash/Desktop/Intent classification/token/tf_model.h5" 
model = TFDebertaV2ForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)

# Define intent mapping
intents = {
    'contact_customer_service': 0, 'review': 1, 'check_invoices': 2, 'recover_password': 3, 
    'switch_account': 4, 'get_invoice': 5, 'complaint': 6, 'payment_issue': 7, 
    'check_payment_methods': 8, 'delivery_period': 9, 'change_order': 10, 
    'set_up_shipping_address': 11, 'place_order': 12, 'cancel_order': 13, 
    'track_order': 14, 'delivery_options': 15, 'check_cancellation_fee': 16, 
    'change_shipping_address': 17, 'contact_human_agent': 18, 'track_refund': 19, 
    'create_account': 20, 'registration_problems': 21, 'delete_account': 22, 
    'get_refund': 23, 'newsletter_subscription': 24, 'check_refund_policy': 25, 
    'edit_account': 26
}

# Reverse mapping for easy lookup
intent_names = {v: k for k, v in intents.items()}

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML file for user input

@app.route('/classify_intent', methods=['POST'])
def classify_intent():
    # Parse JSON request
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'Input text cannot be empty'}), 400

    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="tf", max_length=512)

    # Get predictions from the model
    logits = model(**inputs).logits
    probabilities = tf.nn.softmax(logits, axis=-1)
    predicted_intent_id = tf.argmax(probabilities, axis=-1).numpy()[0]

    # Get intent name
    predicted_intent_name = intent_names[predicted_intent_id]

    return jsonify({
        'intent': predicted_intent_name,
        'confidence': float(probabilities[0][predicted_intent_id])
    })

if __name__ == '__main__':
    app.run(debug=True)

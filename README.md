# Intent Classification using DeBERTa with Flask and Gradio

This project demonstrates the use of **DeBERTa (Decoding-enhanced BERT with disentangled attention)** for **Intent Classification**. The trained model is served through a **Flask** API and integrated with **Gradio** to provide a simple, interactive UI for real-time intent classification from user input.

## Project Overview

The goal of this project is to classify user intents in natural language text using the DeBERTa transformer model. The model is fine-tuned on a custom dataset of text and intent pairs. After training, it is deployed using **Flask** as a RESTful API, and **Gradio** is used to create a user-friendly interface.

### Key Features:
- **DeBERTa-based Intent Classification**: Fine-tuning DeBERTa for accurate NLP-based intent classification.
- **Flask API**: A lightweight REST API for serving the trained model and processing user input.
- **Gradio Interface**: A simple, web-based UI for making predictions without writing code.
- **Real-time Predictions**: Users can classify intents through an intuitive interface.

## Usage

### Step 1: Prepare Your Dataset

Your dataset should consist of text samples and their corresponding intent labels. Here's an example format:

```json
[
    {"text": "Book a flight to Paris", "intent": "Book Flight"},
    {"text": "What's the weather like in New York?", "intent": "Weather Inquiry"},
    ...
]

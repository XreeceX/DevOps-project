#!/bin/bash

echo "🔄 Installing dependencies..."
pip install -r requirements.txt

echo "🚀 Running Fraud Detection Model..."
python src/Fraud_Detection.py

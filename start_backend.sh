#!/bin/bash

echo "Starting Tax RAG Chatbot Backend..."
echo "===================================="

cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.deps_installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.deps_installed
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Creating .env.example..."
    echo "OPENAI_API_KEY=your_api_key_here" > .env
    echo "Please edit .env and add your OpenAI API key (optional)"
fi

# Check if CSV files exist
if [ ! -f "../sam_before.csv" ]; then
    echo "Error: sam_before.csv not found in parent directory!"
    exit 1
fi

if [ ! -f "../taxguru_articles.csv" ]; then
    echo "Warning: taxguru_articles.csv not found in parent directory!"
fi

echo "Starting server..."
python3 main.py


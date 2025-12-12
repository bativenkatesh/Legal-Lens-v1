# Tax RAG Chatbot - Income Tax Act 1961

A Retrieval-Augmented Generation (RAG) based chatbot for querying the Income Tax Act, 1961. This application helps users find relevant tax sections, understand provisions, identify tax-saving opportunities, and get context-aware answers.

## Features

- 🔍 **Semantic Search**: Uses vector embeddings to find the most relevant tax sections
- 🤖 **AI-Powered Responses**: Generates context-aware answers using GPT-4 (with fallback)
- 📚 **Section Retrieval**: Shows relevant sections with similarity scores
- 💡 **Tax Strategy Discovery**: Helps identify legitimate tax-saving opportunities
- 💬 **Conversational Interface**: Maintains conversation history for better context
- 🎨 **Modern UI**: Beautiful React frontend with real-time chat

## Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   React     │  ─────> │   FastAPI    │  ─────> │  Embeddings │
│  Frontend   │         │   Backend    │         │   + LLM     │
└─────────────┘         └──────────────┘         └─────────────┘
                              │
                              ▼
                        ┌─────────────┐
                        │ ExportData  │
                        │    .json    │
                        └─────────────┘
```

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- OpenAI API key (optional, for enhanced responses)

## Installation

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

5. Ensure `ExportData.json` is in the parent directory:
```bash
# The backend expects ExportData.json at ../ExportData.json
# Make sure it exists
ls ../ExportData.json
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

### Start Backend

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

The backend will start on `http://localhost:8000`

**First run will:**
- Load all 931 sections from ExportData.json
- Generate embeddings (takes 2-3 minutes)
- Cache embeddings for faster subsequent runs

### Start Frontend

```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:3000`

## Usage

1. Open `http://localhost:3000` in your browser
2. Ask questions like:
   - "What are the deductions under section 80C?"
   - "How can I save tax on house property?"
   - "What are the penalties for late filing?"
   - "Tell me about tax exemptions for startups"
   - "What are the tax-saving opportunities for salaried employees?"

3. The chatbot will:
   - Find relevant sections using semantic search
   - Generate context-aware responses
   - Show relevant sections with similarity scores
   - Maintain conversation context

## API Endpoints

### POST `/chat`
Main chat endpoint with RAG

**Request:**
```json
{
  "message": "What are section 80C deductions?",
  "conversation_history": []
}
```

**Response:**
```json
{
  "response": "Section 80C provides deductions...",
  "relevant_sections": [
    {
      "section": "80C",
      "title": "Deduction in respect of life insurance premia...",
      "summary": "...",
      "similarity_score": 0.95
    }
  ],
  "confidence": 0.92
}
```

### GET `/search?query=deductions&top_k=10`
Search sections by query

### GET `/section/{section_id}`
Get full details of a specific section

### GET `/health`
Health check endpoint

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── App.css         # Styles
│   │   ├── main.jsx        # Entry point
│   │   └── index.css       # Global styles
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── ExportData.json          # Tax sections dataset
└── README.md
```

## How RAG Works

1. **Embedding Generation**: All sections are converted to vector embeddings using Sentence Transformers
2. **Query Processing**: User query is converted to an embedding
3. **Semantic Search**: Cosine similarity finds the most relevant sections
4. **Context Building**: Top K sections are used as context
5. **Response Generation**: LLM generates answer based on context (or fallback response)

## Features Explained

### Semantic Search
- Uses `all-MiniLM-L6-v2` model for embeddings
- Finds sections based on meaning, not just keywords
- Returns top 5 most relevant sections

### AI Responses
- Uses GPT-4 when OpenAI API key is available
- Falls back to rule-based responses if API key is missing
- Maintains conversation history for context

### Tax Strategy Discovery
- Identifies legitimate tax-saving opportunities
- Explains deductions, exemptions, and provisions
- Highlights conditions and limits

## Troubleshooting

### Backend Issues

**Embeddings not loading:**
- Ensure ExportData.json exists in parent directory
- Check file permissions
- First run takes 2-3 minutes to generate embeddings

**OpenAI API errors:**
- Check API key is set correctly
- Verify API key has credits
- System will use fallback responses if API fails

### Frontend Issues

**CORS errors:**
- Ensure backend is running on port 8000
- Check CORS settings in main.py

**API connection errors:**
- Verify backend is running
- Check API_BASE_URL in App.jsx

## Performance

- **Embedding Generation**: ~2-3 minutes (first run only)
- **Query Response**: ~1-3 seconds (with OpenAI API)
- **Fallback Response**: <1 second
- **Memory Usage**: ~500MB (with embeddings loaded)

## Training Your Own LLM

You can fine-tune a local LLM on the Income Tax Act data:

### Quick Training

```bash
cd backend
pip install -r requirements_training.txt

# Train with default settings (small model, 3 epochs)
python train_llm.py

# Train with custom settings
python train_llm.py \
    --model "microsoft/DialoGPT-small" \
    --output ./tax_llm_model \
    --epochs 5 \
    --batch-size 4 \
    --lr 5e-5
```

### Using Fine-Tuned Model

1. Set environment variable:
```bash
export USE_LOCAL_LLM=true
```

2. Start backend - it will automatically use your fine-tuned model:
```bash
python main.py
```

### Recommended Models for Training

- **Small (Fast, Less Memory)**: `microsoft/DialoGPT-small`, `gpt2`
- **Medium (Better Quality)**: `microsoft/DialoGPT-medium`, `distilgpt2`
- **Large (Best Quality, Needs GPU)**: `gpt2-large`, `EleutherAI/gpt-neo-125M`

### Training Tips

- **GPU Recommended**: Training is much faster on GPU
- **Start Small**: Begin with small models to test
- **Monitor Memory**: Adjust batch_size if you run out of memory
- **More Epochs**: 3-5 epochs usually sufficient for fine-tuning
- **LoRA**: Use PEFT/LoRA for more efficient training (saves memory)

## Future Enhancements

- [ ] Add support for multiple LLM providers (Anthropic, Cohere)
- [ ] Implement document chunking for better retrieval
- [ ] Add citation links to official documents
- [ ] Support for PDF export of conversations
- [ ] Advanced filtering (by chapter, schedule, etc.)
- [ ] Multi-language support

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.


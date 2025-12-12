"""
Fine-tune a local LLM on Income Tax Act data
Uses Hugging Face Transformers for training
"""

import json
import os
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

def load_tax_data():
    """Load and prepare tax data for training"""
    json_path = Path(__file__).parent.parent / "ExportData.json"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} tax sections")
    return data

def create_training_dataset(data, max_length=512):
    """Create training dataset from tax sections"""
    training_texts = []
    
    for item in data:
        section = item.get('section', '')
        title = item.get('title', '')
        content = item.get('content', '')
        summary = item.get('ai_generated_summary', '')
        
        # Create training prompt format
        # Format: Question about section -> Answer with details
        prompt = f"""### Income Tax Act 1961 - Section {section}

Title: {title}

Question: What is Section {section} about and what are its key provisions?

Answer: {summary}

Detailed Provisions:
{content[:1000]}

---

"""
        training_texts.append(prompt)
    
    return training_texts

def prepare_dataset(texts, tokenizer, max_length=512):
    """Tokenize and prepare dataset for training"""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
    
    dataset = Dataset.from_dict({'text': texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    return tokenized_dataset

def train_model(
    model_name="microsoft/DialoGPT-small",  # Small model for local training
    output_dir="./tax_llm_model",
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5
):
    """Train/fine-tune the LLM"""
    
    print("="*70)
    print("FINE-TUNING LLM ON INCOME TAX ACT DATA")
    print("="*70)
    
    # Load tax data
    print("\n1. Loading tax data...")
    tax_data = load_tax_data()
    
    # Create training texts
    print("2. Creating training dataset...")
    training_texts = create_training_dataset(tax_data)
    print(f"   Created {len(training_texts)} training examples")
    
    # Load tokenizer and model
    print(f"\n3. Loading model: {model_name}...")
    print("   (This may take a few minutes for first download)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    print("4. Preparing dataset...")
    dataset = prepare_dataset(training_texts, tokenizer, max_length=512)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="no",
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),  # Use FP16 if GPU available
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n5. Starting training...")
    print("   This will take a while depending on your hardware...")
    trainer.train()
    
    # Save model
    print("\n6. Saving fine-tuned model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ Training complete! Model saved to: {output_dir}")
    print("\nYou can now use this model in the RAG system!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune LLM on Tax Data')
    parser.add_argument(
        '--model',
        type=str,
        default='microsoft/DialoGPT-small',
        help='Base model to fine-tune (default: microsoft/DialoGPT-small)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./tax_llm_model',
        help='Output directory for fine-tuned model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size (default: 4)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='Learning rate (default: 5e-5)'
    )
    
    args = parser.parse_args()
    
    train_model(
        model_name=args.model,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


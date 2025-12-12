"""
Local LLM integration for Tax RAG Chatbot
Uses fine-tuned model or base model for inference
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional

class LocalLLM:
    def __init__(self, model_path: Optional[str] = None, base_model: str = "microsoft/DialoGPT-small"):
        """
        Initialize local LLM
        
        Args:
            model_path: Path to fine-tuned model (if available)
            base_model: Base model to use if fine-tuned model not found
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.base_model = base_model
        
        print(f"Loading LLM on {self.device}...")
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                print(f"Loading fine-tuned model from {self.model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            else:
                print(f"Loading base model: {self.base_model}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
                self.model = AutoModelForCausalLM.from_pretrained(self.base_model)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            # Add padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✓ Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to rule-based responses")
            self.model = None
            self.tokenizer = None
    
    def generate_response(
        self,
        query: str,
        context: str,
        max_length: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response using local LLM
        
        Args:
            query: User query
            context: Context from relevant sections
            max_length: Maximum response length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated response
        """
        if self.model is None or self.tokenizer is None:
            return self._fallback_response(query, context)
        
        # Build prompt
        prompt = f"""### Income Tax Act 1961 - Expert Assistant

Context from relevant sections:
{context}

User Question: {query}

Expert Answer:"""
        
        try:
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer (everything after "Expert Answer:")
            if "Expert Answer:" in generated_text:
                answer = generated_text.split("Expert Answer:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            # Clean up
            answer = answer.split("\n\n")[0]  # Take first paragraph
            answer = answer.split("---")[0]  # Remove separators
            
            return answer if answer else self._fallback_response(query, context)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._fallback_response(query, context)
    
    def _fallback_response(self, query: str, context: str) -> str:
        """Fallback response when model fails"""
        return f"Based on the Income Tax Act, 1961, I found relevant information. Please refer to the context provided above for detailed information about: {query}"

# Global instance
_local_llm = None

def get_local_llm(model_path: Optional[str] = None) -> LocalLLM:
    """Get or create local LLM instance"""
    global _local_llm
    
    if _local_llm is None:
        # Check for fine-tuned model
        default_path = os.path.join(os.path.dirname(__file__), "tax_llm_model")
        if model_path is None and os.path.exists(default_path):
            model_path = default_path
        
        _local_llm = LocalLLM(model_path=model_path)
    
    return _local_llm


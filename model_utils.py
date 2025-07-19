import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import logging
import os
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToSQLModel:
    """Text-to-SQL model wrapper for deployment"""
    
    def __init__(self, model_dir="./final-model", base_model="Salesforce/codet5-base"):
        self.model_dir = model_dir
        self.base_model = base_model
        self.max_length = 128
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer with optimizations for HF Spaces"""
        try:
            # Check if model directory exists
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"Model directory {self.model_dir} not found")
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                use_fast=True
            )
            
            logger.info("Loading base model...")
            # Use lower precision and CPU if needed for memory optimization
            device = "cpu"  # Force CPU for HF Spaces stability
            torch_dtype = torch.float32  # Use float32 for better compatibility
            
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info("Loading PEFT model...")
            self.model = PeftModel.from_pretrained(
                base_model, 
                self.model_dir,
                torch_dtype=torch_dtype,
                device_map=device
            )
            
            # Move to CPU and set to eval mode
            self.model = self.model.to(device)
            self.model.eval()
            
            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Clean up on error
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise
    
    def predict(self, question: str, table_headers: list) -> str:
        """
        Generate SQL query for a given question and table headers
        
        Args:
            question (str): Natural language question
            table_headers (list): List of table column names
            
        Returns:
            str: Generated SQL query
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model not properly loaded")
            
            # Format input text
            table_headers_str = ", ".join(table_headers)
            input_text = f"### Table columns:\n{table_headers_str}\n### Question:\n{question}\n### SQL:"
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.max_length
            )
            
            # Generate prediction with memory optimization
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_length=self.max_length,
                    num_beams=1,  # Use greedy decoding for speed
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode prediction
            sql_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return sql_query.strip()
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            raise
    
    def batch_predict(self, queries: list) -> list:
        """
        Generate SQL queries for multiple questions
        
        Args:
            queries (list): List of dicts with 'question' and 'table_headers' keys
            
        Returns:
            list: List of generated SQL queries
        """
        results = []
        for query in queries:
            try:
                sql = self.predict(query['question'], query['table_headers'])
                results.append({
                    'question': query['question'],
                    'table_headers': query['table_headers'],
                    'sql': sql,
                    'status': 'success'
                })
            except Exception as e:
                logger.error(f"Error in batch prediction for query '{query['question']}': {str(e)}")
                results.append({
                    'question': query['question'],
                    'table_headers': query['table_headers'],
                    'sql': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def health_check(self) -> bool:
        """Check if model is loaded and ready"""
        return (self.model is not None and 
                self.tokenizer is not None and 
                hasattr(self.model, 'generate'))

# Global model instance
_model_instance = None

def get_model():
    """Get or create global model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = TextToSQLModel()
    return _model_instance 
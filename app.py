from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
import time
import os
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None
model_loading = False
model_load_error = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, model_loading, model_load_error
    logger.info("Starting Text-to-SQL API...")
    
    # Start model loading in background
    model_loading = True
    model_load_error = None
    
    try:
        # Import here to avoid startup delays
        from model_utils import get_model
        
        # Set a timeout for model loading (5 minutes)
        try:
            # Run model loading in a thread to avoid blocking
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(get_model)
                model = future.result(timeout=300)  # 5 minute timeout
            logger.info("Model loaded successfully!")
        except concurrent.futures.TimeoutError:
            logger.error("Model loading timed out after 5 minutes")
            model_load_error = "Model loading timed out"
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            model_load_error = str(e)
            
    except Exception as e:
        logger.error(f"Failed to import model_utils: {str(e)}")
        model_load_error = f"Import error: {str(e)}"
    finally:
        model_loading = False
    
    yield
    # Shutdown
    logger.info("Shutting down Text-to-SQL API...")

# Create FastAPI app
app = FastAPI(
    title="Text-to-SQL API",
    description="API for converting natural language questions to SQL queries",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for request/response
class SQLRequest(BaseModel):
    question: str
    table_headers: List[str]

class SQLResponse(BaseModel):
    question: str
    table_headers: List[str]
    sql_query: str
    processing_time: float

class BatchRequest(BaseModel):
    queries: List[SQLRequest]

class BatchResponse(BaseModel):
    results: List[SQLResponse]
    total_queries: int
    successful_queries: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_loading: bool
    model_error: Optional[str] = None
    timestamp: float

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Text-to-SQL API</h1>
                <p>index.html not found. Please ensure the file exists in the same directory.</p>
            </body>
        </html>
        """)

@app.get("/api", response_model=dict)
async def api_info():
    """API information endpoint"""
    return {
        "message": "Text-to-SQL API",
        "version": "1.0.0",
        "endpoints": {
            "/": "GET - Web interface",
            "/api": "GET - API information",
            "/predict": "POST - Generate SQL from single question",
            "/batch": "POST - Generate SQL from multiple questions",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.post("/predict", response_model=SQLResponse)
async def predict_sql(request: SQLRequest):
    """
    Generate SQL query from a natural language question
    
    Args:
        request: SQLRequest containing question and table headers
        
    Returns:
        SQLResponse with generated SQL query
    """
    global model, model_loading, model_load_error
    
    if model_loading:
        raise HTTPException(status_code=503, detail="Model is still loading, please try again in a few minutes")
    
    if model is None:
        error_msg = model_load_error or "Model not loaded"
        raise HTTPException(status_code=503, detail=f"Model not available: {error_msg}")
    
    start_time = time.time()
    
    try:
        sql_query = model.predict(request.question, request.table_headers)
        processing_time = time.time() - start_time
        
        return SQLResponse(
            question=request.question,
            table_headers=request.table_headers,
            sql_query=sql_query,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")

@app.post("/batch", response_model=BatchResponse)
async def batch_predict(request: BatchRequest):
    """
    Generate SQL queries from multiple questions
    
    Args:
        request: BatchRequest containing list of questions and table headers
        
    Returns:
        BatchResponse with generated SQL queries
    """
    global model, model_loading, model_load_error
    
    if model_loading:
        raise HTTPException(status_code=503, detail="Model is still loading, please try again in a few minutes")
    
    if model is None:
        error_msg = model_load_error or "Model not loaded"
        raise HTTPException(status_code=503, detail=f"Model not available: {error_msg}")
    
    start_time = time.time()
    
    try:
        # Convert to format expected by model
        queries = [
            {"question": q.question, "table_headers": q.table_headers}
            for q in request.queries
        ]
        
        # Get predictions
        results = model.batch_predict(queries)
        
        # Convert to response format
        sql_responses = []
        successful_count = 0
        
        for i, result in enumerate(results):
            if result['status'] == 'success':
                successful_count += 1
                sql_responses.append(SQLResponse(
                    question=result['question'],
                    table_headers=result['table_headers'],
                    sql_query=result['sql'],
                    processing_time=time.time() - start_time
                ))
            else:
                # For failed queries, return error in SQL field
                sql_responses.append(SQLResponse(
                    question=result['question'],
                    table_headers=result['table_headers'],
                    sql_query=f"ERROR: {result.get('error', 'Unknown error')}",
                    processing_time=time.time() - start_time
                ))
        
        return BatchResponse(
            results=sql_responses,
            total_queries=len(request.queries),
            successful_queries=successful_count
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in batch prediction: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse with service status
    """
    global model, model_loading, model_load_error
    
    model_loaded = model is not None and model.health_check()
    
    if model_loaded:
        status = "healthy"
    elif model_loading:
        status = "loading"
    else:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_loading=model_loading,
        model_error=model_load_error,
        timestamp=time.time()
    )

@app.get("/example")
async def get_example():
    """Get example usage"""
    return {
        "example_request": {
            "question": "How many employees are older than 30?",
            "table_headers": ["id", "name", "age", "department", "salary"]
        },
        "example_response": {
            "question": "How many employees are older than 30?",
            "table_headers": ["id", "name", "age", "department", "salary"],
            "sql_query": "SELECT COUNT(*) FROM table WHERE age > 30",
            "processing_time": 0.5
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
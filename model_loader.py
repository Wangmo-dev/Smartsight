import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

def load_models():
    """Load all model components"""
    model_path = "saved_models"
    
    # 1. Configure Gemini - PUT YOUR REAL API KEY HERE
    GEMINI_API_KEY = "AIzaSyAgb3fJx7eeyk007dN2IuUYq__4Dyp4r2Q"  # Replace if needed
    genai.configure(api_key=GEMINI_API_KEY)
    
    # 2. Load Sentence Transformer
    try:
        embedder = SentenceTransformer(os.path.join(model_path, "embedder"))
    except Exception as e:
        print(f"Error loading SentenceTransformer: {e}")
        print("Falling back to downloading 'all-MiniLM-L6-v2'")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Load processed data
    try:
        df = pd.read_pickle(os.path.join(model_path, "data.pkl"))
    except Exception as e:
        print(f"Error loading data: {e}")
        df = None
    
    return df, embedder

# Initialize models at startup
df_model, sentence_model = load_models()

# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from deepseek_api import DeepSeekAPI  # Your custom wrapper

# def load_models():
#     # 1. Load saved SentenceTransformer
#     embedder = SentenceTransformer('saved_models/embedder')
    
#     # 2. Load your travel data
#     df = pd.read_pickle('saved_models/data.pkl')
    
#     # 3. Initialize DeepSeek
#     deepseek = DeepSeekAPI(api_key="sk-bf8636f1efdd4365b08325523f2f334b")
    
#     return df, embedder, deepseek

# df_model, sentence_model, deepseek_model = load_models()
import os 
import certifi 
from dotenv import load_dotenv

# --- AUTOMATIC SSL FIX FOR CONDA/PYTHON ENVIRONMENTS ---
ssl_cert = os.environ.get('SSL_CERT_FILE')
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ['SSL_CERT_FILE'] = certifi.where()
elif not ssl_cert:
    os.environ['SSL_CERT_FILE'] = certifi.where()
# --------------------------------------------------------

load_dotenv()
from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    groq_api_key: str = ''
    pinecone_api_key: str = ''
    tavily_api_key: str = ''
    pinecone_index: str = 'medical-voice-chatbot'
    groq_chat_model: str = 'openai/gpt-oss-20b'
    groq_vision_model: str = 'qwen/qwen3.8-27b'
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    api_token: str = ''
    allowed_origins: str = 'http://localhost:8000,http://127.0.0.1:8000'
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    groq_stt_model: str = 'whisper-large-v3-turbo'
    groq_tts_model: str = 'canopylabs/orpheus-v1-english'
    groq_tts_voice: str = 'hannah'



settings = Settings()
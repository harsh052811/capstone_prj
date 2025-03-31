import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.model.google import Gemini
import google.generativeai as genai
from dotenv import load_dotenv
import os
import PyPDF2
from docx import Document
import io
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import pickle
from googleapiclient.http import MediaIoBaseDownload
from openai import RateLimitError, OpenAIError
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Vector search imports
import uuid
import time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Add these imports at the top
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Update imports at the top
from agno.agent import Agent as AgnoAgent
from agno.tools.yfinance import YFinanceTools

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="💬",
    layout="wide"
)

# Style and branding
st.markdown(
    """
    <style>
    .main-header {
        color: #2E4057;
        font-size: 2.5rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .user-message {
        background-color: #F0F2F6;
    }
    .assistant-message {
        background-color: #E8F0FE;
    }
    .message-content {
        width: 90%;
    }
    .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 16px;
        font-weight: bold;
    }
    .user-avatar {
        background-color: #4B9CD3;
        color: white;
    }
    .assistant-avatar {
        background-color: #13B287;
        color: white;
    }
    .stTextInput>div>div>input {
        padding: 0.75rem;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Constants
TESTING_FOLDER_NAME = "testing"  # The specific folder to search in
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks

# =====================
# API CONFIGURATION
# =====================

# Configure OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in your environment variables")
    st.stop()

# Configure Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    st.error("Please set PINECONE_API_KEY in your environment variables")
    st.stop()

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    PINECONE_REGION = "us-east-1"  # As specified
except Exception as e:
    st.error(f"Error initializing Pinecone: {str(e)}")
    st.stop()

# =====================
# UTILITY FUNCTIONS
# =====================

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def get_agent():
    return Agent(
        name="Document Chat Assistant",
        model=Gemini(id="gemini-1.5-pro"),  # Switch back to Gemini
        markdown=True,
    )

def get_user_vector_indexes():
    """Get list of user's vector indexes (vector spaces)"""
    try:
        return pc.list_indexes().names()
    except Exception as e:
        st.error(f"Error getting vector indexes: {str(e)}")
        return []

def init_session_state():
    """Initialize session state variables"""
    if 'current_tool' not in st.session_state:
        st.session_state.current_tool = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processing_query' not in st.session_state:
        st.session_state.processing_query = False
    if 'vector_indexes' not in st.session_state:
        st.session_state.vector_indexes = get_user_vector_indexes()
    if 'current_index' not in st.session_state:
        st.session_state.current_index = None
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = {}

def extract_text_from_pdf(file_obj):
    pdf_reader = PyPDF2.PdfReader(file_obj)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_obj):
    doc = Document(file_obj)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_file(file_obj, mime_type):
    try:
        if 'pdf' in mime_type:
            return extract_text_from_pdf(file_obj)
        elif 'document' in mime_type or 'docx' in mime_type:
            return extract_text_from_docx(file_obj)
        elif 'text/plain' in mime_type:
            return file_obj.getvalue().decode('utf-8')
        else:
            st.error(f"Unsupported file format: {mime_type}")
            return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def create_text_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks for better context preservation"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Try to find a sensible break point (newline or period followed by space)
        if end < text_length:
            # Look for a newline
            newline_pos = text.rfind('\n', start, end)
            if newline_pos > start + chunk_size // 2:  # Only use if it's not too far back
                end = newline_pos + 1
            else:
                # Look for period followed by space
                period_pos = text.rfind('. ', start, end)
                if period_pos > start + chunk_size // 2:  # Only use if it's not too far back
                    end = period_pos + 2
        
        # Extract the chunk
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position for next chunk (with overlap)
        start = end - chunk_overlap if end < text_length else text_length
    
    return chunks

def get_embeddings(chunks, embedding_model):
    """Generate embeddings for text chunks"""
    return embedding_model.encode(chunks)

def create_or_get_index(index_name, dimension=768):
    """Create a new Pinecone index or get an existing one"""
    # Check if the index already exists
    if index_name not in pc.list_indexes().names():
        # Create a new index with new API
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_REGION
            )
        )
        time.sleep(1)  # Give Pinecone time to initialize the index
    
    # Get the index with new API
    return pc.Index(index_name)

def upsert_document_to_pinecone(doc_text, doc_name, index_name=None):
    """Process a document and store its chunks in Pinecone"""
    # Get the embedding model
    model = get_embedding_model()
    dimension = model.get_sentence_embedding_dimension()
    
    # If no index_name provided, create one from document name
    if not index_name:
        # Format document name to be compatible with Pinecone (only lowercase alphanumeric and hyphens)
        index_name = ''.join(e for e in doc_name.lower() if e.isalnum() or e == '-')
        # Remove file extension if present
        index_name = index_name.rsplit('.', 1)[0]
        # Ensure the name starts with a letter (Pinecone requirement)
        if not index_name[0].isalpha():
            index_name = 'doc-' + index_name
    
    # Create or get the index
    index = create_or_get_index(index_name, dimension)
    
    # Split text into chunks
    chunks = create_text_chunks(doc_text)
    
    # Get embeddings
    embeddings = get_embeddings(chunks, model)
    
    # Prepare data for Pinecone - format changed in new API
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Create a unique ID for each chunk
        vector_id = f"{doc_name.replace(' ', '_')}_{uuid.uuid4()}"
        
        # Prepare metadata
        metadata = {
            "text": chunk,
            "document": doc_name,
            "chunk_id": i
        }
        
        # Add to vectors list - new format for vectors
        vectors.append({
            "id": vector_id,
            "values": embedding.tolist(),
            "metadata": metadata
        })
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
    
    # Return information about the uploaded document
    return {
        "document_name": doc_name,
        "chunks": len(chunks),
        "index": index_name
    }

def query_pinecone(query_text, index_name, k=5):
    """Query Pinecone for relevant document chunks"""
    # Get the embedding model
    model = get_embedding_model()
    
    # Generate embedding for the query
    query_embedding = model.encode(query_text).tolist()
    
    # Query the index with new API format
    index = pc.Index(index_name)
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    
    return results

def get_gdrive_service():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If credentials don't exist or are invalid, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)

def find_testing_folder(service):
    """Find the 'testing' folder in Google Drive"""
    query = f"name='{TESTING_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'"
    results = service.files().list(
        q=query,
        spaces='drive',
        fields="files(id, name)"
    ).execute()
    
    folders = results.get('files', [])
    if not folders:
        return None
    
    # Return the first matching folder
    return folders[0]['id']

def list_files_in_folder(service, folder_id, mime_types=None):
    """List files from a specific folder in Google Drive with optional MIME type filtering"""
    query = f"'{folder_id}' in parents"
    
    if mime_types:
        mime_query_parts = [f"mimeType='{mime_type}'" for mime_type in mime_types]
        mime_query = " or ".join(mime_query_parts)
        query += f" and ({mime_query})"
    
    results = service.files().list(
        q=query,
        pageSize=30,
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    
    return results.get('files', [])

def download_file(service, file_id):
    """Download a file from Google Drive by its ID"""
    request = service.files().get_media(fileId=file_id)
    file_content = io.BytesIO()
    downloader = MediaIoBaseDownload(file_content, request)
    done = False
    
    while not done:
        status, done = downloader.next_chunk()
    
    file_content.seek(0)
    return file_content

def extract_text_from_website(url):
    """Extract and clean text content from a website"""
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return None, "Please enter a valid URL (e.g., https://example.com)"
        
        # Fetch webpage
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Extract text
        text = soup.get_text(separator='\n')
        
        # Clean text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = '\n'.join(lines)
        
        return cleaned_text, None
    except requests.RequestException as e:
        return None, f"Error fetching website: {str(e)}"
    except Exception as e:
        return None, f"Error processing website: {str(e)}"

def summarize_text(text):
    """Generate a summary using the AI model"""
    try:
        prompt = f"""
        Please provide a comprehensive summary of the following text. 
        Include the main points and key takeaways. Be concise but thorough.

        Text to summarize:
        {text}

        Instructions:
        1. Start with a brief overview
        2. List main points
        3. Include key takeaways
        4. Use bullet points where appropriate
        5. Keep it clear and concise
        """
        
        agent = get_agent()
        response = agent.run(prompt)
        return response.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data_cached(ticker, period="1y"):
    """Cached version of get_stock_data to reduce API calls"""
    return get_stock_data(ticker, period)

def get_stock_data(ticker, period="1y"):
    """Get stock data using yfinance with rate limit handling"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        
        # Check if we got valid data
        if hist.empty or not info:
            return None, None, "No data available. This could be due to rate limiting. Please try again in a few minutes."
            
        return hist, info, None
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
            return None, None, "Rate limit exceeded. Yahoo Finance is temporarily limiting requests. Please try again in a few minutes."
        return None, None, f"Error fetching stock data: {error_msg}"

def get_stock_data_alternative(ticker):
    """Alternative stock data source if yfinance is rate limited"""
    try:
        # Example using Alpha Vantage API or another source
        # This would require additional imports and possibly API keys
        # ...
        return data, info, None
    except Exception as e:
        return None, None, f"Alternative data source error: {str(e)}"

def analyze_stock(ticker, period="1y"):
    """Analyze stock and return insights"""
    # Use fallback mechanism to get data
    hist, info, error = get_stock_data_with_fallback(ticker, period)
    
    if error:
        return None, error
    
    if hist.empty:
        return None, f"No data found for ticker: {ticker}"
    
    # Basic stock information
    company_name = info.get('shortName', ticker)
    sector = info.get('sector', 'Unknown')
    industry = info.get('industry', 'Unknown')
    market_cap = info.get('marketCap', 0)
    if market_cap > 0:
        market_cap_str = f"${market_cap/1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.2f}M"
    else:
        market_cap_str = "Unknown"
    
    # Calculate key metrics
    current_price = hist['Close'].iloc[-1]
    prev_price = hist['Close'].iloc[0]
    percent_change = ((current_price - prev_price) / prev_price) * 100
    
    # Calculate 50 and 200 day moving averages
    if len(hist) >= 50:
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
    if len(hist) >= 200:
        hist['MA200'] = hist['Close'].rolling(window=200).mean()
    
    # Create a dict with analysis results
    analysis = {
        'ticker': ticker,
        'company_name': company_name,
        'sector': sector,
        'industry': industry,
        'market_cap': market_cap_str,
        'current_price': current_price,
        'change_percent': percent_change,
        'period': period,
        'hist': hist,
        'volume': hist['Volume'].mean(),
        'high_52w': hist['High'].max(),
        'low_52w': hist['Low'].min(),
    }
    
    return analysis, None

def create_stock_chart(analysis):
    """Create an interactive stock chart using Plotly"""
    hist = analysis['hist']
    ticker = analysis['ticker']
    company_name = analysis['company_name']
    
    # Create figure
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name='Price'
    ))
    
    # Add moving averages if available
    if 'MA50' in hist.columns:
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['MA50'],
            name='50-Day MA',
            line=dict(color='orange', width=2)
        ))
    
    if 'MA200' in hist.columns:
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['MA200'],
            name='200-Day MA',
            line=dict(color='purple', width=2)
        ))
    
    # Customize chart
    fig.update_layout(
        title=f'{company_name} ({ticker}) Stock Price',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def generate_stock_insights(analysis):
    """Generate insights about the stock using the AI model"""
    try:
        # Format information for the model
        ticker = analysis['ticker']
        company_name = analysis['company_name']
        current_price = analysis['current_price']
        percent_change = analysis['change_percent']
        market_cap = analysis['market_cap']
        sector = analysis['sector']
        industry = analysis['industry']
        volume = analysis['volume']
        high_52w = analysis['high_52w']
        low_52w = analysis['low_52w']
        period = analysis['period']
        
        # Determine trend
        trend = "upward" if percent_change > 0 else "downward"
        
        # Calculate distance from 52-week high/low
        pct_from_high = ((high_52w - current_price) / high_52w) * 100
        pct_from_low = ((current_price - low_52w) / low_52w) * 100
        
        prompt = f"""
        Please provide a comprehensive analysis of {company_name} ({ticker}) stock based on the following information:
        
        Basic Information:
        - Current Price: ${current_price:.2f}
        - {period} Change: {percent_change:.2f}% ({trend} trend)
        - Market Cap: {market_cap}
        - Sector: {sector}
        - Industry: {industry}
        
        Key Metrics:
        - Average Daily Volume: {volume:.0f}
        - 52-Week High: ${high_52w:.2f} (currently {pct_from_high:.2f}% below)
        - 52-Week Low: ${low_52w:.2f} (currently {pct_from_low:.2f}% above)
        
        Please include:
        1. A brief overview of the company and its current stock performance
        2. Analysis of recent price movements and potential factors
        3. Key considerations for investors based on the given metrics
        4. Any notable patterns or indicators from the price and volume data
        
        Your analysis should be balanced, insightful, and presented in a clear, organized format with appropriate markdown.
        """
        
        agent = get_agent()
        response = agent.run(prompt)
        return response.content
    except Exception as e:
        return f"Error generating stock insights: {str(e)}"

# Add this function to your utility functions section
def get_stock_analysis_with_agno(ticker):
    """Get comprehensive stock analysis using agno"""
    try:
        # Initialize the agno agent with yfinance tools
        # Use only the parameters that are supported by YFinanceTools
        agent = AgnoAgent(
            tools=[YFinanceTools(
                stock_price=True, 
                analyst_recommendations=True, 
                stock_fundamentals=True
            )],
            show_tool_calls=False,
            description="You are an investment analyst that researches stocks thoroughly.",
            instructions=[
                "Format your response using markdown and use tables to display data where possible.",
                "Organize information into clear sections.",
                "Include both technical and fundamental analysis.",
                "Make your analysis accessible to both beginner and experienced investors."
            ],
        )
        
        # Generate the analysis
        prompt = f"""
        Provide a comprehensive analysis of {ticker} stock including:
        1. Current stock price and recent price movements
        2. Key company information and business overview
        3. Stock fundamentals (P/E ratio, market cap, etc.)
        4. Analyst recommendations and price targets
        5. Future outlook and considerations for investors
        
        Organize this information clearly with headings and use tables where appropriate.
        """
        
        response = agent.generate_response(prompt)
        return response, None
    except Exception as e:
        return None, f"Error analyzing stock: {str(e)}"

def get_yahoo_finance_data_alternative(ticker, period="1y"):
    """Alternative implementation using direct requests to Yahoo Finance API"""
    try:
        # Define the periods to map to proper API parameters
        period_map = {
            "1mo": "1m",
            "3mo": "3m",
            "6mo": "6m",
            "1y": "1y",
            "2y": "2y",
            "5y": "5y"
        }
        
        interval = "1d"  # Daily data
        period_param = period_map.get(period, "1y")
        
        # Fetch data from Yahoo Finance API
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={period_param}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
            return None, None, f"No data found for ticker: {ticker}"
        
        # Extract price data
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        
        # Create dataframe
        df = pd.DataFrame({
            'Open': quote.get('open', []),
            'High': quote.get('high', []),
            'Low': quote.get('low', []),
            'Close': quote.get('close', []),
            'Volume': quote.get('volume', [])
        }, index=[datetime.fromtimestamp(ts) for ts in timestamps])
        
        # Get company info from a different endpoint
        info_url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
        info_response = requests.get(info_url, headers=headers)
        info_data = info_response.json()
        
        info = {}
        if "quoteResponse" in info_data and "result" in info_data["quoteResponse"] and info_data["quoteResponse"]["result"]:
            quote_info = info_data["quoteResponse"]["result"][0]
            info = {
                'shortName': quote_info.get('shortName', ticker),
                'sector': quote_info.get('sector', 'Unknown'),
                'industry': quote_info.get('industry', 'Unknown'),
                'marketCap': quote_info.get('marketCap', 0)
            }
        
        return df, info, None
    except Exception as e:
        return None, None, f"Error fetching data: {str(e)}"

def get_alpha_vantage_data(ticker, period="1y"):
    """Get stock data using Alpha Vantage API"""
    try:
        # You need to sign up for a free API key at https://www.alphavantage.co/support/#api-key
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return None, None, "Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY in environment variables."
        
        # Map period to appropriate function
        if period in ["1mo", "3mo", "6mo"]:
            function = "TIME_SERIES_DAILY"
            outputsize = "compact"  # last 100 data points
        else:
            function = "TIME_SERIES_DAILY"
            outputsize = "full"  # full history (up to 20 years)
        
        # Make API request for time series data
        url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&outputsize={outputsize}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            return None, None, f"Alpha Vantage API error: {data['Error Message']}"
        
        if "Time Series (Daily)" not in data:
            return None, None, "No data returned from Alpha Vantage"
        
        # Parse the response into a pandas DataFrame
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        
        for date, values in time_series.items():
            df.loc[date] = [
                float(values["1. open"]),
                float(values["2. high"]),
                float(values["3. low"]),
                float(values["4. close"]),
                int(values["5. volume"])
            ]
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Filter by period
        if period == "1mo":
            df = df.last('30D')
        elif period == "3mo":
            df = df.last('90D')
        elif period == "6mo":
            df = df.last('180D')
        elif period == "1y":
            df = df.last('365D')
        elif period == "2y":
            df = df.last('730D')
        elif period == "5y":
            df = df.last('1825D')
        
        # Get company overview
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
        overview_response = requests.get(overview_url)
        overview_data = overview_response.json()
        
        info = {
            'shortName': overview_data.get('Name', ticker),
            'sector': overview_data.get('Sector', 'Unknown'),
            'industry': overview_data.get('Industry', 'Unknown'),
            'marketCap': float(overview_data.get('MarketCapitalization', 0))
        }
        
        return df, info, None
    except Exception as e:
        return None, None, f"Error fetching Alpha Vantage data: {str(e)}"

def get_stock_data_with_fallback(ticker, period="1y"):
    """Get stock data with fallback mechanisms if primary source fails"""
    
    # Try regular yfinance first (cached)
    hist, info, error = get_stock_data_cached(ticker, period)
    if not error and hist is not None and not hist.empty:
        return hist, info, None
    
    # If yfinance failed with rate limit, try the alternative Yahoo implementation
    if error and ("rate limit" in error.lower() or "too many requests" in error.lower()):
        st.warning("⚠️ YFinance rate limited, trying alternative Yahoo Finance implementation...")
        hist, info, error = get_yahoo_finance_data_alternative(ticker, period)
        if not error and hist is not None and not hist.empty:
            return hist, info, None
    
    # If both Yahoo methods failed, try Alpha Vantage if available
    if os.getenv("ALPHA_VANTAGE_API_KEY"):
        st.warning("⚠️ Yahoo Finance methods failed, trying Alpha Vantage API...")
        hist, info, error = get_alpha_vantage_data(ticker, period)
        if not error and hist is not None and not hist.empty:
            return hist, info, None
    
    # All methods failed
    return None, None, "Unable to fetch stock data from any available source. Please try again later."

# =====================
# MAIN APPLICATION
# =====================

# Define SCOPES before we try to use it in get_gdrive_service
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Initialize session state
init_session_state()

# Initialize the agent
doc_agent = get_agent()

# Tool selector section (add this at the beginning of the main UI, before sidebar)
st.markdown("# 🤖 Smart Assistant")

# Tool selector
tool_options = {
    "Document Chat": "Chat with your uploaded documents",
    "Website Summarizer": "Get summaries of web pages",
    "Stock Market Analysis": "Analyze stocks and get AI insights"
}

selected_tool = st.radio(
    "Select a tool:",
    options=list(tool_options.keys()),
    format_func=lambda x: f"{x} - {tool_options[x]}"
)

st.session_state.current_tool = selected_tool

# Connect to Google Drive (regardless of tool)
if 'drive_service' not in st.session_state:
    try:
        st.session_state.drive_service = get_gdrive_service()
    except Exception as e:
        st.error(f"Error connecting to Google Drive: {str(e)}")
        st.info("Please ensure you have credentials.json file with proper Google Drive API credentials")
        st.stop()

# Find the testing folder only if Document Chat is selected
if st.session_state.current_tool == "Document Chat" and 'testing_folder_id' not in st.session_state:
    try:
        folder_id = find_testing_folder(st.session_state.drive_service)
        if folder_id:
            st.session_state.testing_folder_id = folder_id
        else:
            st.error(f"Folder '{TESTING_FOLDER_NAME}' not found in your Google Drive")
            st.info(f"Please create a folder named '{TESTING_FOLDER_NAME}' in your Google Drive and upload your documents there")
    except Exception as e:
        st.error(f"Error finding the testing folder: {str(e)}")

# Then structure the main application based on the selected tool
if st.session_state.current_tool == "Website Summarizer":
    st.markdown("## 🌐 Website Summarizer")
    
    with st.form("website_form"):
        url = st.text_input("Enter website URL:", placeholder="https://example.com")
        submit_url = st.form_submit_button("Summarize")
    
    if submit_url and url:
        with st.spinner("Analyzing website..."):
            # Extract text from website
            text, error = extract_text_from_website(url)
            
            if error:
                st.error(error)
            elif text:
                # Generate summary
                with st.spinner("Generating summary..."):
                    summary = summarize_text(text)
                    
                    # Display results in an expander
                    with st.expander("Website Summary", expanded=True):
                        st.markdown(summary)
                    
                    # Option to view original text
                    with st.expander("View Original Text", expanded=False):
                        st.text_area("Extracted text:", text, height=300)
            else:
                st.error("No content could be extracted from the website.")

elif st.session_state.current_tool == "Document Chat":
    # Only show the sidebar for document chat
    with st.sidebar:
        st.header("Document Management")
        
        # Vector Space (Index) Management
        st.subheader("Vector Space")
        
        # Refresh the list of vector indexes
        if st.button("Refresh Vector Spaces"):
            st.session_state.vector_indexes = get_user_vector_indexes()
            st.success("Vector spaces refreshed!")
        
        # Create a new vector space
        with st.expander("Create New Vector Space"):
            new_index_name = st.text_input("Vector Space Name (lowercase, no spaces)", key="new_index")
            if st.button("Create Vector Space"):
                if new_index_name:
                    try:
                        # Format index name to be compatible with Pinecone (only lowercase alphanumeric and hyphens)
                        formatted_name = ''.join(e for e in new_index_name.lower() if e.isalnum() or e == '-')
                        # Get model dimension
                        model = get_embedding_model()
                        dimension = model.get_sentence_embedding_dimension()
                        
                        create_or_get_index(formatted_name, dimension)
                        st.session_state.vector_indexes = get_user_vector_indexes()
                        st.success(f"Vector space '{formatted_name}' created!")
                    except Exception as e:
                        st.error(f"Error creating vector space: {str(e)}")
                else:
                    st.warning("Please enter a name for the vector space")
        
        # Select an existing vector space
        if st.session_state.vector_indexes:
            st.session_state.current_index = st.selectbox(
                "Select Vector Space", 
                options=st.session_state.vector_indexes,
                key="index_selector"
            )
            
            # Show document upload only if a vector space is selected
            if st.session_state.current_index:
                st.divider()
                st.subheader("Upload Documents")
                st.markdown(f"Files from your '{TESTING_FOLDER_NAME}' folder:")
                
                # Define supported file types
                supported_mime_types = [
                    'application/pdf',
                    'application/vnd.google-apps.document',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'text/plain'
                ]
                
                try:
                    files = list_files_in_folder(st.session_state.drive_service, st.session_state.testing_folder_id, supported_mime_types)
                    
                    if not files:
                        st.info(f"No compatible documents found in your '{TESTING_FOLDER_NAME}' folder. Please upload PDF, DOCX, or TXT files to this folder.")
                    else:
                        file_options = {f"{file['name']}": file for file in files}
                        selected_file_option = st.selectbox("Choose a document", options=list(file_options.keys()))
                        selected_file = file_options[selected_file_option]
                        
                        if st.button("Process and Upload Document"):
                            with st.spinner(f"Processing {selected_file['name']}..."):
                                file_content = download_file(st.session_state.drive_service, selected_file['id'])
                                extracted_text = extract_text_from_file(file_content, selected_file['mimeType'])
                                
                                if extracted_text:
                                    # Upload to Pinecone
                                    result = upsert_document_to_pinecone(
                                        extracted_text, 
                                        selected_file['name'],
                                        index_name=None
                                    )
                                    
                                    # Refresh vector indexes list
                                    st.session_state.vector_indexes = get_user_vector_indexes()
                                    
                                    # Store info about uploaded document
                                    index_name = result['index']
                                    if index_name not in st.session_state.uploaded_documents:
                                        st.session_state.uploaded_documents[index_name] = []
                                        
                                    st.session_state.uploaded_documents[index_name].append(result)
                                    
                                    st.success(f"Document processed and uploaded to new vector space '{index_name}'!")
                                else:
                                    st.error("Failed to extract text from the selected document")
                except Exception as e:
                    st.error(f"Error accessing Google Drive: {str(e)}")
                
                # Display uploaded documents
                if st.session_state.current_index in st.session_state.uploaded_documents and st.session_state.uploaded_documents[st.session_state.current_index]:
                    st.divider()
                    st.subheader("Uploaded Documents")
                    for doc in st.session_state.uploaded_documents[st.session_state.current_index]:
                        st.markdown(f"📄 **{doc['document_name']}** - {doc['chunks']} chunks")
        else:
            st.info("No vector spaces found. Please create one to get started.")
    
    # Main chat UI for document chat
    if st.session_state.current_index:
        st.markdown("## 💬 Document Chat")
        st.markdown(f"**Current Vector Space:** {st.session_state.current_index}")
        
        # Add clear chat button
        if st.button("Clear Chat History"):
            # Add a welcome message
            welcome_message = {
                "role": "assistant", 
                "content": f"👋 Hi there! I'm your document assistant for the '{st.session_state.current_index}' vector space. How can I help you today?"
            }
            st.session_state.messages = [welcome_message]
            st.rerun()
        
        # Initialize chat with welcome message if empty
        if not st.session_state.messages:
            welcome_message = {
                "role": "assistant", 
                "content": f"👋 Hi there! I'm your document assistant for the '{st.session_state.current_index}' vector space. How can I help you today?"
            }
            st.session_state.messages = [welcome_message]
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="avatar user-avatar">👤</div>
                        <div class="message-content">{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div class="avatar assistant-avatar">💬</div>
                        <div class="message-content">{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # User input
        # Create a form to better control submission
        with st.form(key="question_form", clear_on_submit=True):
            user_input = st.text_input("Type your question here", key="user_input", placeholder="What would you like to know about these documents?")
            submit_button = st.form_submit_button("Send")
        
        # Handle form submission
        if submit_button and user_input and not st.session_state.processing_query:
            try:
                # Set processing flag to prevent multiple runs
                st.session_state.processing_query = True
                
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Generate response
                with st.spinner("Thinking..."):
                    # Create conversation history string
                    conversation_history = ""
                    for msg in st.session_state.messages[:-1]:  # Exclude the latest user message
                        role = "User" if msg["role"] == "user" else "Assistant"
                        conversation_history += f"{role}: {msg['content']}\n\n"
                        
                    # Query Pinecone for relevant document chunks
                    search_results = query_pinecone(
                        user_input, 
                        st.session_state.current_index,
                        k=5
                    )
                    
                    # Extract and format relevant document chunks
                    document_context = ""
                    if 'matches' in search_results and search_results['matches']:
                        for i, match in enumerate(search_results['matches']):
                            document_context += f"Document: {match['metadata']['document']}\n"
                            document_context += f"Content: {match['metadata']['text']}\n\n"
                    else:
                        document_context = "No relevant document content found."

                    qa_prompt = f"""
                    You are a friendly and helpful customer service representative responding to questions about business documents. 
                    Your tone should be conversational, helpful, and professional.

                    Vector Space: {st.session_state.current_index}
                    
                    Relevant Document Sections:
                    ```
                    {document_context}
                    ```

                    Previous Conversation:
                    {conversation_history}

                    Current Question: {user_input}

                    Instructions:
                    1. Consider the previous conversation context when formulating your response
                    2. Respond in a warm, conversational customer service tone
                    3. Provide a helpful, well-structured answer without citing evidence or references
                    4. Use natural language as if you were chatting with a customer in real-time
                    5. If you can't answer from the document, politely explain what information is available
                    6. DO NOT include "Supporting Evidence" or reference quotes from the document
                    7. DO NOT use phrases like "Based on the document" or "According to the document"
                    8. Format your response with appropriate markdown where helpful

                    Your response should feel like a natural conversation with a friendly customer service representative.
                    """
                    
                    response = doc_agent.run(qa_prompt)
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                # Remove the last user message if there was an error
                if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()
            finally:
                # Reset processing flag
                st.session_state.processing_query = False
                st.rerun()

elif st.session_state.current_tool == "Stock Market Analysis":
    st.markdown("## 📈 Stock Market Analysis")
    
    # Stock input form
    with st.form("stock_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            ticker = st.text_input("Enter Stock Ticker Symbol:", 
                                  placeholder="AAPL, MSFT, GOOGL, AMZN, etc.")
        
        with col2:
            period_options = {
                "1mo": "1 Month", 
                "3mo": "3 Months", 
                "6mo": "6 Months",
                "1y": "1 Year", 
                "2y": "2 Years", 
                "5y": "5 Years"
            }
            period = st.selectbox("Time Period:", 
                                 options=list(period_options.keys()),
                                 format_func=lambda x: period_options[x],
                                 index=3)  # Default to 1y
        
        analyze_button = st.form_submit_button("Analyze Stock")
    
    # Process stock analysis
    if analyze_button and ticker:
        ticker = ticker.upper().strip()
        
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Add a small delay before making requests
                time.sleep(1)  # 1 second delay

                # Try using Agno first
                agno_analysis, agno_error = get_stock_analysis_with_agno(ticker)
                
                if not agno_error and agno_analysis:
                    # Display Agno analysis
                    st.markdown(agno_analysis)
                else:
                    if agno_error and "rate limit" in agno_error.lower():
                        st.warning("⚠️ Rate limit reached with Agno. Trying alternative analysis method...")
                    
                    # Fall back to custom analysis
                    analysis, error = analyze_stock(ticker, period)
                    
                    if error:
                        if "rate limit" in error.lower() or "too many requests" in error.lower():
                            st.error("📈 Rate Limit Exceeded")
                            st.info("Yahoo Finance is temporarily limiting requests due to high traffic. Please try again in a few minutes or try a different ticker symbol.")
                            
                            # Add a retry button
                            if st.button("Retry Analysis"):
                                st.experimental_set_query_params(ticker=ticker, retry=True)
                                st.rerun()
                        else:
                            st.error(error)
                    elif analysis:
                        # Display current price and change
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                label=f"{analysis['company_name']} ({ticker})",
                                value=f"${analysis['current_price']:.2f}",
                                delta=f"{analysis['change_percent']:.2f}%"
                            )
                        
                        with col2:
                            st.metric(
                                label="Market Cap",
                                value=analysis['market_cap']
                            )
                        
                        with col3:
                            st.metric(
                                label="Sector",
                                value=analysis['sector']
                            )
                        
                        # Display stock chart
                        st.plotly_chart(create_stock_chart(analysis), use_container_width=True)
                        
                        # Generate and display AI insights
                        with st.spinner("Generating insights..."):
                            insights = generate_stock_insights(analysis)
                            
                            with st.expander("📊 Stock Analysis & Insights", expanded=True):
                                st.markdown(insights)
                        
                        # Display additional metrics
                        st.subheader("Additional Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                label="52-Week High",
                                value=f"${analysis['high_52w']:.2f}"
                            )
                        
                        with col2:
                            st.metric(
                                label="52-Week Low",
                                value=f"${analysis['low_52w']:.2f}"
                            )
                        
                        with col3:
                            st.metric(
                                label="Avg Daily Volume",
                                value=f"{analysis['volume']:.0f}"
                            )
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.info("Please try again in a few minutes or try a different ticker symbol.")
    else:
        # Display default information when no ticker is entered
        st.info("👆 Enter a stock ticker symbol above to get started.")
        
        # Example tickers for quick selection
        st.markdown("### Popular Tickers:")
        
        # Create buttons for popular stocks
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("AAPL (Apple)"):
                st.experimental_set_query_params(ticker="AAPL")
                st.rerun()
        
        with col2:
            if st.button("MSFT (Microsoft)"):
                st.experimental_set_query_params(ticker="MSFT")
                st.rerun()
        
        with col3:
            if st.button("NVDA (NVIDIA)"):
                st.experimental_set_query_params(ticker="NVDA")
                st.rerun()

        with col4:
            if st.button("AMZN (Amazon)"):
                st.experimental_set_query_params(ticker="AMZN")
                st.rerun()
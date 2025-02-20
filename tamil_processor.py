from googletrans import Translator
import requests
import re
from typing import List, Dict
import google.generativeai as genai
import time

# Configure APIs
GOOGLE_API_KEY = 'AIzaSyAX9rGWvyzsp6-viAzfhl269LH6zDkDTAI'
JINA_API_KEY = 'jina_c385fc4abd104c10b05c0485d032688baDqQLo7XPAjGwo7l6HdBQawsgOeH'

# Initialize APIs
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')
translator = Translator()

# Jina AI API configuration
JINA_HEADERS = {
    'Authorization': f'Bearer {JINA_API_KEY}',
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}

def clean_tamil_text(text: str) -> str:
    """Clean and normalize Tamil text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    return text

def create_text_chunks(text: str, chunk_size: int = 300) -> list:
    """Split text into chunks of roughly equal size"""
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip() + '.'
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_embedding(text: str) -> List[float]:
    """Generate embeddings using Jina AI API"""
    url = "https://api.jina.ai/v1/embeddings"
    
    try:
        response = requests.post(
            url,
            headers=JINA_HEADERS,
            json={
                "input": text,
                "model": "jina-embeddings-v2-base-en"
            }
        )
        
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        else:
            print(f"Error generating embedding: {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception in generate_embedding: {str(e)}")
        return None

def check_duplicate_content(text: str, index) -> bool:
    """Check if similar content already exists in the vector database"""
    try:
        # Generate embedding for the text
        embedding = generate_embedding(text)
        if not embedding:
            return False
        
        # Query Pinecone with high similarity threshold
        results = index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True,
            include_values=False
        )
        
        if results.matches:
            # Check if similarity score is above threshold (0.95 for near-duplicates)
            if results.matches[0].score > 0.95:
                print(f"Duplicate content found with similarity score: {results.matches[0].score}")
                return True
        
        return False
        
    except Exception as e:
        print(f"Error checking duplicates: {str(e)}")
        return False

def store_tamil_vectors(text: str, title: str, index) -> dict:
    """Process and store Tamil text in vector database"""
    try:
        print("\n=== Processing Tamil Text ===")
        print(f"Title: {title}")
        print(f"Original text length: {len(text)} characters")
        
        # Clean the text
        cleaned_text = clean_tamil_text(text)
        if not cleaned_text:
            return {'success': False, 'error': 'No valid text after cleaning'}
        print(f"Cleaned text length: {len(cleaned_text)} characters")
        
        # Create chunks
        print("\n=== Creating Chunks ===")
        chunks = create_text_chunks(cleaned_text)
        if not chunks:
            return {'success': False, 'error': 'Failed to create text chunks'}
        print(f"Total chunks created: {len(chunks)}")
        
        # Initialize translator
        translator = Translator()
        vectors_to_store = []
        
        print("\n=== Processing Chunks ===")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}/{len(chunks)}:")
            print(f"Length: {len(chunk)} characters")
            print("Content preview:", chunk[:100], "..." if len(chunk) > 100 else "")
            
            # Check for duplicates before processing
            if check_duplicate_content(chunk, index):
                print(f"✗ Chunk {i} skipped (duplicate content)")
                continue
                
            try:
                # Translate Tamil to English for better embedding
                translated = translator.translate(chunk, src='ta', dest='en')
                if not translated or not translated.text:
                    print(f"✗ Chunk {i} translation failed")
                    continue
                print("✓ Translation successful")
                
                # Generate embedding
                embedding = generate_embedding(translated.text)
                if not embedding:
                    print(f"✗ Chunk {i} embedding generation failed")
                    continue
                print("✓ Embedding generated")
                
                # Create safe title for vector ID
                safe_title = re.sub(r'[^a-zA-Z0-9_-]', '', title.lower().replace(' ', '_'))
                if not safe_title:
                    safe_title = 'tamil_text'
                safe_title = safe_title[:30]  # Limit length
                
                vector_id = f"tamil_{safe_title}_{int(time.time())}_{i}"
                
                vectors_to_store.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'title': title,
                        'content': chunk,
                        'translated_content': translated.text,
                        'language': 'tamil',
                        'source': 'tamil_input'
                    }
                })
                print("✓ Vector prepared for storage")
                
            except Exception as e:
                print(f"✗ Error processing chunk {i}: {str(e)}")
                continue
        
        if not vectors_to_store:
            return {'success': False, 'error': 'No vectors to store after processing'}
        
        # Store vectors in batches
        print(f"\n=== Storing Vectors in Pinecone ===")
        batch_size = 10
        total_batches = (len(vectors_to_store) + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, len(vectors_to_store), batch_size), 1):
            batch = vectors_to_store[i:i + batch_size]
            print(f"\nStoring batch {batch_num}/{total_batches} ({len(batch)} vectors)")
            index.upsert(vectors=batch)
            print(f"✓ Batch {batch_num} stored successfully")
        
        print("\n=== Storage Summary ===")
        print(f"Total chunks processed: {len(chunks)}")
        print(f"Vectors stored: {len(vectors_to_store)}")
        print(f"Chunks skipped: {len(chunks) - len(vectors_to_store)}")
        
        return {
            'success': True,
            'chunks_stored': len(vectors_to_store),
            'message': f'Successfully stored {len(vectors_to_store)} chunks'
        }
        
    except Exception as e:
        print(f"\n✗ Error in store_tamil_vectors: {str(e)}")
        return {'success': False, 'error': str(e)}

def get_tamil_response(query: str, index, chat_history: list = None) -> str:
    """Generate response for Tamil queries using vector search"""
    try:
        # Translate query to English for better matching
        translator = Translator()
        translated_query = translator.translate(query, src='ta', dest='en').text
        
        # Generate embedding for the query
        query_embedding = generate_embedding(translated_query)
        if not query_embedding:
            return "Sorry, I couldn't process your query. Please try again."
        
        # Search in vector database
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
        if not results.matches:
            return "I couldn't find any relevant information to answer your question."
        
        # Format context from relevant matches
        context = "Based on the following Tamil content:\n\n"
        for match in results.matches:
            context += f"- {match.metadata['content']}\n"
        
        # Include chat history for context
        if chat_history:
            history_context = "\nPrevious conversation:\n"
            for msg in chat_history[-3:]:
                history_context += f"User: {msg['content']}\nAssistant: {msg['content']}\n"
            context += history_context
        
        # Create prompt for response generation
        prompt = f"""{context}

User question (in Tamil): {query}
Translated question: {translated_query}

Please provide a detailed response in Tamil that:
1. Directly answers the user's question
2. Uses the relevant information from the context
3. Maintains a natural conversational tone
4. Is accurate and helpful

Response should be in Tamil language."""
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error in get_tamil_response: {str(e)}")
        return "நான் உங்கள் கேள்விக்கு பதிலளிக்க முயற்சிக்கும்போது ஒரு பிழை ஏற்பட்டது. தயவுசெய்து மீண்டும் முயற்சிக்கவும்." 
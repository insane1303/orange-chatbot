from PyPDF2 import PdfReader
import re
from typing import List, Dict
import os
import google.generativeai as genai

# Configure Gemini
GOOGLE_API_KEY = 'AIzaSyAX9rGWvyzsp6-viAzfhl269LH6zDkDTAI'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Global variable to store temporary PDF content
temp_pdf_content = {
    'filename': None,
    'total_pages': 0,
    'chunks': [],
    'raw_content': '',
    'content_preview': '',
    'error': None
}

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove special characters but keep sentence structure
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

def preprocess_text(text: str) -> str:
    """Enhanced text preprocessing for better embeddings"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Replace common unicode characters
    text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace('…', '...')
    
    # Normalize sentence endings
    text = re.sub(r'([.!?])\s*(?=[A-Z])', r'\1\n\n', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', '', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,!?:;\-–—()[\]{}"\']', ' ', text)
    
    return text.strip()

def create_text_chunks(text: str, chunk_size: int = 250, overlap: int = 50) -> List[Dict[str, str]]:
    """Split text into overlapping chunks while preserving semantic boundaries"""
    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        # Split paragraph into sentences
        sentences = re.split('(?<=[.!?])\s+', paragraph)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence.split())
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > chunk_size and current_chunk:
                # Create chunk with metadata
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'length': current_length,
                    'sentences': len(current_chunk)
                })
                
                # Keep last sentence for overlap
                overlap_sentences = current_chunk[-1:] if current_chunk else []
                current_chunk = overlap_sentences
                current_length = sum(len(s.split()) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add the last chunk if not empty
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'length': current_length,
            'sentences': len(current_chunk)
        })
    
    return chunks

def extract_pdf_content(file_path: str) -> str:
    """Extract text content from PDF file"""
    global temp_pdf_content
    try:
        # Open the PDF file
        reader = PdfReader(file_path)
        text_content = []
        
        # Store total pages
        temp_pdf_content['total_pages'] = len(reader.pages)
        
        # Extract text from each page
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
        
        # Join all text content
        content = '\n\n'.join(text_content)
        
        if not content.strip():
            error_msg = "No text content found in PDF"
            temp_pdf_content['error'] = error_msg
            raise ValueError(error_msg)
            
        # Store raw content
        temp_pdf_content['raw_content'] = content
        temp_pdf_content['filename'] = os.path.basename(file_path)
        temp_pdf_content['error'] = None
        
        return content
        
    except Exception as e:
        error_msg = f"Error extracting PDF content: {str(e)}"
        print(error_msg)
        temp_pdf_content['error'] = error_msg
        raise

def process_pdf(file_path: str) -> List[Dict[str, str]]:
    """Process PDF file and return chunks with metadata"""
    global temp_pdf_content
    try:
        print(f"\n=== Processing PDF: {file_path} ===")
        # Extract content
        content = extract_pdf_content(file_path)
        print(f"Extracted raw content length: {len(content)}")
        
        # Clean and preprocess content
        cleaned_content = clean_text(content)
        preprocessed_content = preprocess_text(cleaned_content)
        print(f"Preprocessed content length: {len(preprocessed_content)}")
        
        # Generate chunks
        chunks = create_text_chunks(preprocessed_content)
        print(f"Generated {len(chunks)} chunks")
        
        # Store chunks in temporary storage
        temp_pdf_content['chunks'] = chunks
        temp_pdf_content['content_preview'] = preprocessed_content[:500] + '...' if len(preprocessed_content) > 500 else preprocessed_content
        print("Content stored in temporary storage")
        print(f"Preview: {temp_pdf_content['content_preview'][:200]}...")
        
        return chunks
        
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        print(error_msg)
        temp_pdf_content['error'] = error_msg
        raise

def get_temp_pdf_content() -> Dict:
    """Get the temporary stored PDF content"""
    global temp_pdf_content
    return {
        'filename': temp_pdf_content['filename'],
        'total_pages': temp_pdf_content['total_pages'],
        'chunks': temp_pdf_content['chunks'],
        'content_preview': temp_pdf_content['content_preview'],
        'error': temp_pdf_content['error']
    }

def clear_temp_pdf_content():
    """Clear the temporary PDF content storage"""
    global temp_pdf_content
    temp_pdf_content = {
        'filename': None,
        'total_pages': 0,
        'chunks': [],
        'raw_content': '',
        'content_preview': '',
        'error': None
    }

def get_pdf_response(user_message: str, chat_history: list = None) -> str:
    """Get response for PDF-related questions using temporary storage"""
    global temp_pdf_content
    
    try:
        if not temp_pdf_content['chunks']:
            return "Please upload a PDF first!"
            
        # Combine all chunks for context
        chunks_text = [chunk['text'] for chunk in temp_pdf_content['chunks']]
        pdf_context = '\n'.join(chunks_text)
        
        # Format chat history
        history_context = ""
        if chat_history:
            formatted_history = []
            for msg in chat_history[-3:]:  # Include last 3 exchanges
                role = "User" if msg['role'] == 'user' else "Assistant"
                formatted_history.append(f"{role}: {msg['content']}")
            history_context = "\n".join(formatted_history)
        
        prompt = f"""Based ONLY on the following PDF content, answer the user's question. Do NOT use any external knowledge or information not present in the PDF content:

PDF Content:
{pdf_context}

Previous conversation:
{history_context}

User question: {user_message}

Important Instructions:
1. ONLY use information explicitly stated in the PDF content above
2. If the answer cannot be found in the PDF content, say "I cannot find this information in the PDF"
3. Do not make assumptions or add information from general knowledge
4. Use direct quotes from the PDF when possible
5. Format the response using markdown for clarity

Remember: You must ONLY use information from the PDF content. Do not add any external knowledge or assumptions."""

        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error in PDF chat: {str(e)}")
        return "I apologize, but I encountered an error processing your request. Please try again." 
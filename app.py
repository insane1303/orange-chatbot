from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import time
from urllib.parse import urlparse
from werkzeug.utils import secure_filename

# Import functions from our new modules
from website_extractor import preprocess_text, fetch_single_page
from pdf_extractor import process_pdf, extract_pdf_content, preprocess_text, get_temp_pdf_content, get_pdf_response
from tamil_processor import store_tamil_vectors, get_tamil_response
from invoice_processor import process_invoice_image, get_invoice_response
from resume_processor import process_resumes, get_resume_response

app = Flask(__name__)

# Configure Gemini API
GOOGLE_API_KEY = 'AIzaSyAX9rGWvyzsp6-viAzfhl269LH6zDkDTAI'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Initialize sentence transformer model
encoder = SentenceTransformer('intfloat/multilingual-e5-large')  # To match Pinecone's model

# Initialize Pinecone
PINECONE_API_KEY = 'pcsk_7KR89L_7StzjPDfNqfB3xxmaBkEKnsGWuPvMXCzVBQcpRcN2WkmkvNbDUnxkm4Tm4YF3xe'

# Add these configurations after the Flask app initialization
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize chat history storage for each functionality
chat_histories = {
    'normal': {},
    'website': {},
    'pdf': {},
    'tamil': {},
    'invoice': {},
    'resume': {}  # Add resume chat history
}

# Initialize global variables
websiteContent = None

def debug_print(message, data=None):
    """Print debug information"""
    print(f"\n=== DEBUG: {message} ===")
    if data:
        print(data)
    print("=" * (len(message) + 8))

def init_pinecone():
    global index
    try:
        # Initialize Pinecone
        pc = Pinecone(
            api_key=PINECONE_API_KEY
        )
        
        # Create or get index
        INDEX_NAME = "website-content"
        DIMENSION = 768  # Changed to match Jina embeddings dimension

        # List existing indexes
        existing_indexes = pc.list_indexes()
        print(f"Existing indexes: {existing_indexes.names()}")

        # Create index if it doesn't exist
        if INDEX_NAME not in existing_indexes.names():
            print(f"Creating new index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            time.sleep(20)
        
        # Get the index
        print(f"Connecting to index: {INDEX_NAME}")
        index = pc.Index(INDEX_NAME)
        print("Successfully connected to Pinecone")
        return index
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        return None

# Initialize index
print("Starting Pinecone initialization...")
index = init_pinecone()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fetch-website', methods=['POST'])
def fetch_website():
    global websiteContent
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'})
    
    try:
        # Set response timeout
        response_timeout = 30  # 30 seconds timeout
        
        def process_with_timeout():
            try:
                return fetch_single_page(url)
            except Exception as e:
                print(f"Error in process_with_timeout: {str(e)}")
                return {'success': False, 'error': str(e)}
        
        # Run with timeout
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(process_with_timeout)
            try:
                result = future.result(timeout=response_timeout)
            except TimeoutError:
                return jsonify({
                    'success': False,
                    'error': 'Request timed out. The webpage might be too large or slow to respond. Please try again with a different URL.'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Error processing request: {str(e)}'
                })
        
        if not result or not isinstance(result, dict):
            return jsonify({
                'success': False,
                'error': 'Invalid response from website processor'
            })
        
        if not result.get('success', False):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error occurred')
            })
        
        # Store the website content globally
        websiteContent = {
            'url': url,
            'title': result.get('title', ''),
            'debug_content': {
                'url': url,
                'title': result.get('title', ''),
                'content': result.get('content', '')
            }
        }
        
        return jsonify({
            'success': True,
            'url': url,
            'title': result.get('title', ''),
            'chunks_stored': len(result.get('chunks', [])),
            'debug_content': {
                'url': url,
                'title': result.get('title', ''),
                'content': result.get('content', '')[:500] + '...' if len(result.get('content', '')) > 500 else result.get('content', '')
            }
        })
    
    except Exception as e:
        print(f"Error in fetch_website route: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/store-tamil', methods=['POST'])
def store_tamil():
    global index
    
    if index is None:
        return jsonify({'success': False, 'error': 'Pinecone index not initialized'})
    
    data = request.json
    text = data.get('text')
    title = data.get('title', 'Untitled')
    
    if not text:
        return jsonify({'success': False, 'error': 'No text provided'})
    
    try:
        result = store_tamil_vectors(text, title, index)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in store_tamil route: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/upload-invoice', methods=['POST'])
def upload_invoice():
    print("Received invoice upload request")
    
    if 'invoice' not in request.files:
        print("No file in request")
        return jsonify({'success': False, 'error': 'No file uploaded'})

    file = request.files['invoice']
    print(f"Received file: {file.filename}")
    
    try:
        if file.filename == '':
            print("No file selected")
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            print(f"Invalid file type: {file.filename}")
            return jsonify({'success': False, 'error': 'Invalid file type. Only PNG, JPG, and JPEG files are allowed.'})
        
        file_data = file.read()
        result = process_invoice_image(file_data, file.filename)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error uploading invoice: {str(e)}")
        return jsonify({'success': False, 'error': f'Error processing invoice: {str(e)}'})

@app.route('/upload-resumes', methods=['POST'])
def upload_resumes():
    try:
        print("Received resume upload request")
        
        # Get job requirements from form data
        job_requirements = request.form.get('job_requirements')
        if not job_requirements:
            return jsonify({'success': False, 'error': 'No job requirements provided'})
        
        # Check if files were uploaded
        if 'resumes[]' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'})
        
        resume_files = request.files.getlist('resumes[]')
        if not resume_files:
            return jsonify({'success': False, 'error': 'No files selected'})
        
        # Process each resume file
        processed_resumes = []
        for file in resume_files:
            if file.filename == '':
                continue
                
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': f'Invalid file type for {file.filename}. Only PDF files are allowed.'})
            
            # Read file data
            file_data = file.read()
            processed_resumes.append({
                'filename': file.filename,
                'data': file_data
            })
        
        if not processed_resumes:
            return jsonify({'success': False, 'error': 'No valid resume files found'})
        
        # Process resumes
        result = process_resumes(job_requirements, processed_resumes)
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing resumes: {str(e)}")
        return jsonify({'success': False, 'error': f'Error processing resumes: {str(e)}'})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data['message']
    has_website = data.get('hasWebsite', False)
    has_pdf = data.get('hasPDF', False)
    has_tamil = data.get('hasTamil', False)
    has_invoice = data.get('hasInvoice', False)
    has_resume = data.get('hasResume', False)  # Add resume flag
    session_id = data.get('sessionId', 'default')
    
    try:
        # Get the appropriate chat history
        chat_type = 'resume' if has_resume else 'invoice' if has_invoice else 'tamil' if has_tamil else 'pdf' if has_pdf else 'website' if has_website else 'normal'
        
        if session_id not in chat_histories[chat_type]:
            chat_histories[chat_type][session_id] = []
        
        chat_history = chat_histories[chat_type][session_id]
        
        # Store user message
        chat_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Get appropriate response based on type
        if has_resume:
            response = get_resume_response(user_message, chat_history)
        elif has_invoice:
            response = get_invoice_response(user_message, chat_history)
        elif has_tamil:
            response = get_tamil_response(user_message, index, chat_history)
        elif has_pdf:
            response = get_pdf_response(user_message, chat_history)
        elif has_website:
            response = get_website_response(user_message, chat_history)
        else:
            response = get_normal_response(user_message, chat_history)
        
        # Store bot response
        chat_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return jsonify({
            'response': response
        })
        
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return jsonify({
            'response': "I apologize, but I encountered an error processing your request. Please try again."
        })

def format_chat_history(history):
    """Format chat history for prompt context"""
    if not history:
        return ""
    formatted = []
    for msg in history[-3:]:  # Include last 3 exchanges
        role = "User" if msg['role'] == 'user' else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("Received PDF upload request")
        # Check if a file was uploaded
        if 'pdf' not in request.files:
            print("No file in request")
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['pdf']
        print(f"Received file: {file.filename}")
        
        # Check if a file was selected
        if file.filename == '':
            print("No file selected")
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            print(f"Invalid file type: {file.filename}")
            return jsonify({'success': False, 'error': 'Invalid file type. Only PDF files are allowed.'})
        
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {filepath}")
        file.save(filepath)
        
        try:
            # Process PDF and store in temporary storage
            print("Processing PDF content")
            chunks = process_pdf(filepath)
            
            # Get console output data
            console_data = get_temp_pdf_content()
            print(f"Processed PDF: {len(chunks)} chunks, {console_data['total_pages']} pages")
            
            return jsonify({
                'success': True,
                'message': 'PDF processed successfully',
                'filename': filename,
                'total_pages': console_data['total_pages'],
                'content_preview': console_data['content_preview'],
                'chunks': chunks,
                'error': console_data['error']
            })
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return jsonify({'success': False, 'error': f'Error processing PDF content: {str(e)}'})
            
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return jsonify({'success': False, 'error': f'Error uploading file: {str(e)}'})

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return no content for favicon requests

def get_normal_response(user_message: str, chat_history: list = None) -> str:
    """Generate a response for normal chat messages."""
    try:
        # Format chat history if provided
        context = ""
        if chat_history and len(chat_history) > 0:
            context = "Previous conversation:\n"
            for msg in chat_history[-3:]:  # Include last 3 messages for context
                context += f"{msg['role']}: {msg['content']}\n"
        
        prompt = f"""{context}
User Question: {user_message}

Please provide a helpful and detailed response. Format your response using markdown when appropriate."""
            
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating normal response: {str(e)}")
        return "I apologize, but I encountered an error while generating a response. Please try again."

def get_website_response(user_message: str, chat_history: list = None) -> str:
    """Generate a response based on the website content and user message."""
    global websiteContent
    if not websiteContent:
        return "No website content available. Please fetch a website first."
    
    # Format chat history if provided
    context = ""
    if chat_history and len(chat_history) > 0:
        context = "Previous conversation:\n"
        for msg in chat_history[-3:]:  # Include last 3 messages for context
            context += f"User: {msg['content']}\nAssistant: {msg['content']}\n"
    
    prompt = f"""{context}
Website Title: {websiteContent.get('title', 'Unknown')}
Website Content: {websiteContent.get('debug_content', {}).get('content', '')}

User Question: {user_message}

Please provide a detailed response based on the website content above. Format your response using markdown when appropriate."""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error while generating a response. Please try again."

if __name__ == '__main__':
    app.run(debug=True) 
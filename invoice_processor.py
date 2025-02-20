import google.generativeai as genai
from typing import Dict, List
import re
import pytesseract
from PIL import Image
import io
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import os

# Configure Gemini
GOOGLE_API_KEY = 'AIzaSyAX9rGWvyzsp6-viAzfhl269LH6zDkDTAI'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Global variable to store temporary invoice content
temp_invoice_content = {
    'filename': None,
    'raw_text': '',
    'summary': None,
    'extracted_data': None,
    'error': None
}

def extract_text_from_image(image_data: bytes) -> str:
    """Extract text from image using OCR"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(image)
        return text.strip()
        
    except Exception as e:
        print(f"Error in OCR: {str(e)}")
        raise

def clean_invoice_text(text: str) -> str:
    """Clean and normalize invoice text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize common invoice terms
    text = text.replace('inv.', 'invoice')
    text = text.replace('amt.', 'amount')
    text = text.replace('qty.', 'quantity')
    
    return text.strip()

def extract_invoice_data(text: str) -> Dict:
    """Extract key information from invoice text"""
    try:
        # Create prompt for Gemini to extract structured data
        prompt = f"""Extract the following information from this invoice text:
1. Invoice number
2. Date
3. Total amount
4. Company/Vendor name
5. Line items (with quantities and prices if available)
6. Payment terms (if available)
7. Tax details (if available)

Invoice text:
{text}

Format the response as a detailed JSON object with these fields. If any field is not found, set it to null."""

        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error extracting invoice data: {str(e)}")
        raise

def summarize_invoice(text: str) -> str:
    """Generate a natural language summary of the invoice"""
    try:
        prompt = f"""Create a clear and concise summary of this invoice that includes:
1. Who issued it and to whom
2. The total amount and main items/services
3. Key dates and payment terms
4. Any notable details or special conditions

Invoice text:
{text}

Please format the summary in a clear, readable way using markdown."""

        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error summarizing invoice: {str(e)}")
        raise

def extract_text_from_pdf(pdf_data: bytes) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        # Create a file-like object from bytes
        pdf_stream = io.BytesIO(pdf_data)
        
        # Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text_content = []
        
        # Extract text from each page
        for page in doc:
            text_content.append(page.get_text())
        
        # Close the document
        doc.close()
        
        # Join all text content
        text = "\n\n".join(text_content)
        return text.strip()
        
    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
        raise

def extract_text_from_file(file_data: bytes, file_type: str) -> str:
    """Extract text from file based on type"""
    if file_type == 'pdf':
        return extract_text_from_pdf(file_data)
    else:  # image
        return extract_text_from_image(file_data)

def process_invoice_file(file_data: bytes, filename: str) -> Dict:
    """Process invoice file (PDF or image) and return extracted information"""
    global temp_invoice_content
    
    try:
        print(f"\n=== Processing Invoice File: {filename} ===")
        
        # Determine file type
        file_type = 'pdf' if filename.lower().endswith('.pdf') else 'image'
        print(f"File type: {file_type}")
        
        # Extract text based on file type
        raw_text = extract_text_from_file(file_data, file_type)
        if not raw_text:
            raise ValueError("No text could be extracted from the file")
        print("✓ Text extracted from file")
        
        # Clean the extracted text
        cleaned_text = clean_invoice_text(raw_text)
        print("✓ Text cleaned and normalized")
        
        # Extract structured data
        extracted_data = extract_invoice_data(cleaned_text)
        print("✓ Structured data extracted")
        
        # Generate summary
        summary = summarize_invoice(cleaned_text)
        print("✓ Summary generated")
        
        # Store in temporary storage
        temp_invoice_content = {
            'filename': filename,
            'raw_text': raw_text,
            'summary': summary,
            'extracted_data': extracted_data,
            'error': None
        }
        
        return {
            'success': True,
            'filename': filename,
            'summary': summary,
            'extracted_data': extracted_data
        }
        
    except Exception as e:
        error_msg = f"Error processing invoice: {str(e)}"
        print(f"✗ {error_msg}")
        temp_invoice_content['error'] = error_msg
        return {
            'success': False,
            'error': error_msg
        }

# Replace the old process_invoice_image function with process_invoice_file
process_invoice_image = process_invoice_file

def get_invoice_response(query: str, chat_history: list = None) -> str:
    """Generate response for invoice-related questions"""
    global temp_invoice_content
    
    try:
        if not temp_invoice_content['raw_text']:
            return "Please upload an invoice image first!"
        
        # Format context
        context = f"""Invoice Content:
Raw Text: {temp_invoice_content['raw_text']}

Extracted Data: {temp_invoice_content['extracted_data']}

Summary: {temp_invoice_content['summary']}"""

        # Add chat history if available
        if chat_history:
            history_context = "\nPrevious conversation:\n"
            for msg in chat_history[-3:]:
                role = "User" if msg['role'] == 'user' else "Assistant"
                history_context += f"{role}: {msg['content']}\n"
            context += history_context

        prompt = f"""{context}

User question: {query}

Please provide a detailed answer that:
1. Uses specific information from the invoice
2. Is clear and well-formatted using markdown
3. Provides exact numbers and details when available
4. Stays focused on the invoice content

Remember: Only use information from the invoice to answer the question."""

        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error in invoice chat: {str(e)}")
        return "I apologize, but I encountered an error processing your request. Please try again." 
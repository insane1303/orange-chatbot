from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import re
import google.generativeai as genai

# Configure APIs
GOOGLE_API_KEY = 'AIzaSyAX9rGWvyzsp6-viAzfhl269LH6zDkDTAI'

# Initialize APIs
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Temporary storage for website content
website_content = {
    'url': None,
    'title': None,
    'content': None
}

def preprocess_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Replace common unicode characters
    text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace('…', '...')
    
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www.\S+|\S+@\S+', '', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,!?:;\-–—()[\]{}"\']', ' ', text)
    
    return text.strip()

def extract_main_content(soup: BeautifulSoup) -> str:
    """Extract main content from HTML"""
    try:
        print("Starting content extraction...")
        
        # First, remove unwanted elements but keep more potential content
        unwanted_elements = [
            'script', 'style', 'noscript', 'iframe'
        ]
        
        # Remove elements by tag name
        for tag in unwanted_elements:
            for element in soup.find_all(tag):
                element.decompose()
        
        print("Removed unwanted elements")
        
        # Remove only the most problematic classes
        common_unwanted_classes = [
            'cookie', 'popup', 'modal', 'advertisement'
        ]
        
        for class_name in common_unwanted_classes:
            for element in soup.find_all(class_=lambda x: x and class_name.lower() in str(x).lower()):
                element.decompose()
        
        print("Removed unwanted elements by class")
        
        # Extract title for context
        title = ''
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)
            print(f"Found title: {title}")
        
        # Try multiple methods to find content
        main_content = []
        
        # Method 1: Look for main content containers
        print("Method 1: Looking for main containers...")
        for tag in ['main', 'article', '[role="main"]', '#main', '#content', '.main', '.content', '.post', '.entry', '.article']:
            container = soup.select_one(tag)
            if container:
                print(f"Found container with selector: {tag}")
                text = container.get_text(strip=True, separator=' ')
                if len(text) > 100:  # Reduced minimum length
                    main_content = [text]
                    print(f"Found content in container: {text[:100]}...")
                    break
        
        # Method 2: Look for article-like containers
        if not main_content:
            print("Method 2: Looking for article containers...")
            article_containers = soup.find_all(['article', 'div', 'section'])
            
            if article_containers:
                print(f"Found {len(article_containers)} potential containers")
                # Get the container with most text content
                container = max(article_containers, key=lambda x: len(x.get_text(strip=True)))
                text = container.get_text(strip=True, separator=' ')
                if len(text) > 100:  # Reduced minimum length
                    main_content = [text]
                    print(f"Found content in largest container: {text[:100]}...")
        
        # Method 3: Get all paragraphs
        if not main_content:
            print("Method 3: Collecting all paragraphs...")
            paragraphs = []
            for p in soup.find_all(['p', 'div', 'section', 'article']):
                text = p.get_text(strip=True)
                if len(text) > 30:  # Reduced minimum length
                    paragraphs.append(text)
            
            if paragraphs:
                print(f"Found {len(paragraphs)} paragraphs")
                main_content = paragraphs
                print(f"Sample paragraph: {paragraphs[0][:100]}...")
        
        # Method 4: Final fallback - get all text from body
        if not main_content:
            print("Method 4: Final fallback - extracting from body...")
            body = soup.find('body')
            if body:
                # Get all text blocks
                text_blocks = []
                for element in body.stripped_strings:
                    if len(element.strip()) > 20:  # Reduced minimum length
                        text_blocks.append(element.strip())
                if text_blocks:
                    print(f"Found {len(text_blocks)} text blocks")
                    main_content = text_blocks
                    print(f"Sample text block: {text_blocks[0][:100]}...")
        
        if main_content:
            # Join all content with title
            content = '\n\n'.join([title] + main_content if title else main_content)
            # Clean the content
            content = preprocess_text(content)
            
            if len(content) > 50:  # Reduced minimum length
                print(f"Successfully extracted content ({len(content)} characters)")
                print(f"Content preview: {content[:200]}...")
                return content
            else:
                print(f"Extracted content too short: {len(content)} characters")
        else:
            print("No main content found in any method")
        
        # Last resort: just get all text from the page
        print("Last resort: getting all text from page...")
        all_text = soup.get_text(strip=True, separator=' ')
        if len(all_text) > 50:
            print(f"Found {len(all_text)} characters of text")
            return preprocess_text(all_text)
        
        return ""
        
    except Exception as e:
        print(f"Error extracting content: {str(e)}")
        return ""

def fetch_single_page(url: str) -> dict:
    """Fetch and process a single webpage"""
    global website_content
    
    try:
        # Check if we already have this URL
        if website_content['url'] == url:
            return {
                'success': True,
                'url': url,
                'title': website_content['title'],
                'content': website_content['content']
            }
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        print(f"Fetching URL: {url}")
        
        # Request headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        # Fetch the page
        response = requests.get(url, headers=headers, timeout=(3, 10))
        response.raise_for_status()
        
        # Check if content is HTML
        if 'text/html' not in response.headers.get('Content-Type', '').lower():
            return {'success': False, 'error': 'Not an HTML page'}
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else ''
        
        # Extract and clean content
        content = extract_main_content(soup)
        if not content:
            return {'success': False, 'error': 'No content found'}
        
        # Store in temporary variable
        website_content = {
            'url': url,
            'title': title,
            'content': content
        }
        
        return {
            'success': True,
            'url': url,
            'title': title,
            'content': content
        }
        
    except requests.Timeout:
        return {'success': False, 'error': 'Request timed out'}
    except requests.RequestException as e:
        return {'success': False, 'error': f'Error fetching page: {str(e)}'}
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {'success': False, 'error': str(e)}

def get_website_response(user_message: str) -> str:
    """Get response for website queries"""
    try:
        if not website_content['content']:
            return "Please fetch a website first."
        
        prompt = f"""Based on the following webpage content: {website_content['content'][:1000]}...

User question: {user_message}

Please provide a detailed answer that:
1. Uses specific information from the webpage content
2. Is clear and well-formatted
3. Stays focused on the webpage's content
4. Uses markdown formatting when helpful

Remember: Only use information from the webpage content to answer the question."""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error in website chat: {str(e)}")
        return "I apologize, but I encountered an error processing your request. Please try again." 
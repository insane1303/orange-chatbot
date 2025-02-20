import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, List, Any
import asyncio
from urllib.parse import urljoin
import re
from concurrent.futures import ThreadPoolExecutor
import logging
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import json

class SinglePageScraper:
    def __init__(self):
        self.session = self._create_session()
        self.timeout = (30, 30)  # (connect, read) timeouts
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
    def _create_session(self) -> requests.Session:
        """Create a session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
        
    def extract_content(self, url: str) -> Dict[str, Any]:
        """Extract content from a single webpage with thorough parsing"""
        try:
            # Fetch page with timeout and retry handling
            response = self.session.get(
                url, 
                headers=self.headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content
            content = {
                'title': self._extract_title(soup),
                'meta_description': self._extract_meta_description(soup),
                'main_content': self._extract_main_content(soup),
                'headings': self._extract_headings(soup),
                'lists': self._extract_lists(soup),
                'tables': self._extract_tables(soup),
                'links': self._extract_links(soup, url),
                'images': self._extract_images(soup, url),
                'structured_data': self._extract_structured_data(soup),
                'text_blocks': self._extract_text_blocks(soup)
            }
            
            return {
                'success': True,
                'url': url,
                'content': content,
                'error': None
            }
            
        except requests.Timeout:
            return {
                'success': False,
                'url': url,
                'content': None,
                'error': 'Request timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'url': url,
                'content': None,
                'error': str(e)
            }
            
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title = soup.title.string if soup.title else ''
        h1 = soup.find('h1')
        h1_text = h1.get_text(strip=True) if h1 else ''
        return title or h1_text or ''
        
    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description"""
        meta = soup.find('meta', {'name': 'description'})
        return meta['content'] if meta else ''
        
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content with priority areas"""
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
            
        # Try to find main content area
        main_content = ''
        priority_tags = ['article', 'main', '[role="main"]', '.main-content', '#main-content']
        
        for tag in priority_tags:
            content = soup.select(tag)
            if content:
                main_content = ' '.join(
                    element.get_text(strip=True, separator=' ')
                    for element in content
                )
                break
                
        # Fallback to content-like divs if no main content found
        if not main_content:
            content_divs = soup.find_all('div', class_=re.compile(r'content|article|post'))
            main_content = ' '.join(
                div.get_text(strip=True, separator=' ')
                for div in content_divs
            )
            
        return main_content
        
    def _extract_headings(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract hierarchical headings"""
        headings = []
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            for heading in soup.find_all(tag):
                headings.append({
                    'level': tag,
                    'text': heading.get_text(strip=True)
                })
        return headings
        
    def _extract_lists(self, soup: BeautifulSoup) -> List[List[str]]:
        """Extract lists from content"""
        lists = []
        for list_tag in soup.find_all(['ul', 'ol']):
            items = [
                item.get_text(strip=True)
                for item in list_tag.find_all('li')
                if item.get_text(strip=True)
            ]
            if items:
                lists.append(items)
        return lists
        
    def _extract_tables(self, soup: BeautifulSoup) -> List[List[List[str]]]:
        """Extract tables from content"""
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                row_data = [
                    cell.get_text(strip=True)
                    for cell in row.find_all(['td', 'th'])
                ]
                if row_data:
                    table_data.append(row_data)
            if table_data:
                tables.append(table_data)
        return tables
        
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract relevant links"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            if href and text:
                full_url = urljoin(base_url, href)
                links.append({
                    'url': full_url,
                    'text': text
                })
        return links
        
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract image information"""
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            if src:
                full_url = urljoin(base_url, src)
                images.append({
                    'url': full_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', '')
                })
        return images
        
    def _extract_structured_data(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract structured data (JSON-LD, microdata)"""
        structured_data = []
        
        # Extract JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                structured_data.append(data)
            except:
                continue
                
        return structured_data
        
    def _extract_text_blocks(self, soup: BeautifulSoup) -> List[str]:
        """Extract meaningful text blocks"""
        text_blocks = []
        
        for element in soup.find_all(['p', 'div']):
            text = element.get_text(strip=True)
            if len(text) > 100:  # Only keep substantial blocks
                text_blocks.append(text)
                
        return text_blocks

# Usage example
async def scrape_webpage(url: str) -> Dict[str, Any]:
    scraper = SinglePageScraper()
    result = scraper.extract_content(url)
    
    if result['success']:
        print(f"Successfully scraped {url}")
        return result['content']
    else:
        print(f"Error scraping {url}: {result['error']}")
        return None 
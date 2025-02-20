import google.generativeai as genai
from typing import Dict, List
import fitz  # PyMuPDF
import io
import re
import os
from datetime import datetime
import json

# Configure Gemini
GOOGLE_API_KEY = 'AIzaSyAX9rGWvyzsp6-viAzfhl269LH6zDkDTAI'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Global variable to store resumes and job requirements
temp_resume_data = {
    'job_requirements': None,
    'resumes': [],
    'analysis': None,
    'rankings': None,
    'error': None
}

def extract_links_from_pdf(doc):
    """Extract hyperlinks from PDF"""
    links = []
    for page in doc:
        for link in page.get_links():
            if 'uri' in link:
                links.append({
                    'url': link['uri'],
                    'page': page.number + 1
                })
    return links

def extract_personal_details(text: str) -> Dict:
    """Extract personal details from resume text"""
    try:
        prompt = f"""Extract personal details from this resume text. Include:
1. Full Name
2. Email
3. Phone Number
4. Location/Address
5. LinkedIn Profile (if present)
6. Other Social/Portfolio Links
7. Current Position/Title (if present)

Resume Text:
{text}

Format the response as a JSON object with these fields. If a field is not found, set it to null."""

        response = model.generate_content(prompt)
        return eval(response.text)  # Convert string response to dict
    except Exception as e:
        print(f"Error extracting personal details: {str(e)}")
        return {}

def extract_text_from_pdf(pdf_data: bytes) -> tuple:
    """Extract text and metadata from PDF using PyMuPDF"""
    try:
        # Create a file-like object from bytes
        pdf_stream = io.BytesIO(pdf_data)
        
        # Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text_content = []
        
        # Extract text from each page
        for page in doc:
            text_content.append(page.get_text())
        
        # Extract hyperlinks
        links = extract_links_from_pdf(doc)
        
        # Join all text content
        text = "\n\n".join(text_content)
        
        # Close the document
        doc.close()
        
        return text.strip(), links
        
    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
        raise

def clean_resume_text(text: str) -> str:
    """Clean and normalize resume text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s.,@()\-:;/]', ' ', text)
    
    # Normalize common terms
    text = text.replace('e mail', 'email')
    text = text.replace('E mail', 'email')
    
    return text.strip()

def analyze_resume(resume_text: str, job_requirements: str, personal_details: Dict, links: List) -> Dict:
    """Analyze a single resume against job requirements"""
    try:
        prompt = f"""Analyze this resume based on the following requirements and provide a detailed assessment.

Job Requirements/Instructions:
{job_requirements}

Personal Details:
{personal_details}

Resume:
{resume_text}

Links found in resume:
{links}

Please analyze the resume following the job requirements/instructions exactly as specified. 
Provide the analysis in the following JSON format:
{{
    "match_score": <number between 0-100 based on how well the resume matches the requirements>,
    "key_skills_match": [<list of relevant skills found in resume that match requirements>],
    "experience_relevance": "<detailed analysis of experience as per requirements>",
    "education_fit": "<analysis of education based on requirements>",
    "strengths": [<list of strengths relevant to the requirements>],
    "gaps": [<list of missing requirements or areas of improvement>],
    "overall_assessment": "<detailed assessment focusing on the specific requirements>",
    "personal_details_formatted": "<nicely formatted personal details>",
    "links_and_profiles": [<list of relevant links and profiles>],
    "custom_analysis": "<additional analysis based on any specific instructions in the requirements>"
}}

Ensure the analysis strictly follows the provided requirements/instructions."""

        response = model.generate_content(prompt)
        # Parse the response text to ensure it's valid JSON
        try:
            # First try to parse as JSON
            analysis_dict = json.loads(response.text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to clean and evaluate the string
            cleaned_response = response.text.strip().replace('\n', ' ').replace('```json', '').replace('```', '')
            analysis_dict = eval(cleaned_response)
        
        # Add personal details and links to the analysis
        analysis_dict['personal_details'] = personal_details
        analysis_dict['links'] = links
        
        return analysis_dict
        
    except Exception as e:
        print(f"Error analyzing resume: {str(e)}")
        # Return a structured dictionary if analysis fails
        return {
            'match_score': 0,
            'key_skills_match': [],
            'experience_relevance': 'Unable to analyze experience',
            'education_fit': 'Unable to analyze education',
            'strengths': [],
            'gaps': ['Analysis failed'],
            'overall_assessment': 'Error in analysis',
            'personal_details_formatted': str(personal_details),
            'links_and_profiles': [str(link) for link in links],
            'personal_details': personal_details,
            'links': links,
            'custom_analysis': 'Analysis failed'
        }

def rank_resumes(analyses: List[Dict]) -> str:
    """Generate a ranking and comparison of all resumes"""
    try:
        # Format analyses text with proper name extraction
        analyses_text = []
        for i, analysis in enumerate(analyses):
            name = analysis.get('personal_details', {}).get('Full Name', f'Applicant {i+1}')
            # Create a clean summary of the analysis
            summary = f"""
Candidate: {name}
Match Score: {analysis.get('match_score', 'N/A')}
Key Skills: {', '.join(analysis.get('key_skills_match', []))}
Experience: {analysis.get('experience_relevance', 'N/A')}
Education: {analysis.get('education_fit', 'N/A')}
Strengths: {', '.join(analysis.get('strengths', []))}
Gaps: {', '.join(analysis.get('gaps', []))}
Overall Assessment: {analysis.get('overall_assessment', 'N/A')}
"""
            analyses_text.append(summary)
        
        analyses_text = "\n---\n".join(analyses_text)
        
        prompt = f"""Based on these resume analyses, provide a detailed comparison and ranking:

{analyses_text}

Please provide a detailed analysis in the following format:
1. Ranked list of candidates from best to worst match, including their match scores
2. Key strengths and differentiators for each candidate
3. Specific areas where each candidate excels or needs improvement
4. Recommendations for each candidate
5. Justification for the ranking order

Format the response using clear markdown headings and bullet points."""

        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error ranking resumes: {str(e)}")
        return "Error occurred while ranking resumes. Please try again."

def process_resumes(job_requirements: str, resume_files: List[Dict]) -> Dict:
    """Process multiple resumes and compare them against job requirements"""
    global temp_resume_data
    
    try:
        print(f"\n=== Processing {len(resume_files)} Resumes ===")
        
        # Store job requirements
        temp_resume_data['job_requirements'] = job_requirements
        temp_resume_data['resumes'] = []
        temp_resume_data['analysis'] = []
        
        # Process each resume
        for resume_file in resume_files:
            file_data = resume_file['data']
            filename = resume_file['filename']
            
            print(f"\nProcessing resume: {filename}")
            
            try:
                # Extract text and links
                raw_text, links = extract_text_from_pdf(file_data)
                cleaned_text = clean_resume_text(raw_text)
                
                # Extract personal details
                personal_details = extract_personal_details(raw_text)
                
                # Analyze resume
                analysis = analyze_resume(cleaned_text, job_requirements, personal_details, links)
                
                # Store resume data
                temp_resume_data['resumes'].append({
                    'filename': filename,
                    'text': raw_text,  # Store raw text to preserve formatting
                    'personal_details': personal_details,
                    'links': links,
                    'analysis': analysis
                })
                
                print(f"✓ Processed resume for {personal_details.get('Full Name', filename)}")
                
            except Exception as e:
                print(f"Error processing resume {filename}: {str(e)}")
                continue
        
        if not temp_resume_data['resumes']:
            return {
                'success': False,
                'error': 'No resumes were successfully processed'
            }
        
        # Rank all resumes
        analyses = [resume['analysis'] for resume in temp_resume_data['resumes']]
        rankings = rank_resumes(analyses)
        temp_resume_data['rankings'] = rankings
        
        return {
            'success': True,
            'message': f'Successfully processed {len(temp_resume_data["resumes"])} resumes',
            'rankings': rankings
        }
        
    except Exception as e:
        error_msg = f"Error processing resumes: {str(e)}"
        print(f"✗ {error_msg}")
        temp_resume_data['error'] = error_msg
        return {
            'success': False,
            'error': error_msg
        }

def get_resume_response(query: str, chat_history: list = None) -> str:
    """Generate response for resume-related questions"""
    global temp_resume_data
    
    try:
        if not temp_resume_data['resumes']:
            return "Please upload resumes and set job requirements first!"
        
        # Format context with detailed information
        context = f"""Job Requirements:
{temp_resume_data['job_requirements']}

Resume Rankings and Analysis:
{temp_resume_data['rankings']}

Individual Resume Details:
"""
        for resume in temp_resume_data['resumes']:
            context += f"""
### {resume['personal_details'].get('Full Name', 'Unknown Applicant')}
Personal Details:
{resume['personal_details']}

Links found in resume:
{resume['links']}

Full Resume Text:
{resume['text']}

Analysis:
{resume['analysis']}

---
"""

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
1. Uses specific information from the resumes and analysis
2. Includes relevant personal details and links when asked
3. Is clear and well-formatted using markdown
4. Provides specific examples and comparisons when relevant
5. Stays focused on the resume content and job requirements
6. Uses applicant names when referring to specific resumes

Remember: Only use information from the provided context to answer the question."""

        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error in resume chat: {str(e)}")
        return "I apologize, but I encountered an error processing your request. Please try again." 
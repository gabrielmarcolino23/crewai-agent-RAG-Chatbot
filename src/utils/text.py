from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter

def process_files(files):
    text = ''
    
    for file in files:
        pdf = PdfReader(file)
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    
    return text

def create_text_chunks(text):
  
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(text)  
    
    return chunks

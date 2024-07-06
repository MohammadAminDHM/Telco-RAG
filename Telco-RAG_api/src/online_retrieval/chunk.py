import re

def custom_text_splitter(text, chunk_size, word_split=False):

    chunks = []
    start = 0
    separators_pattern = re.compile(r'[\s,.\-!?\[\]\(\){}":;<>]+')
    
    while start < len(text) - 25:
        end = min(start + chunk_size, len(text))
        
        if word_split:
            match = separators_pattern.search(text, end)
            if match:
                end = match.end()
                
        if end == start:
            end = start + 1
            
        chunks.append(text[start:end])
        start = end - 25
        
        if word_split:
            match = separators_pattern.search(text, start-1)
            if match:
                start = match.start() + 1
                
        if start < 0:
            start = 0
    chunked_doc =[]
    for chunk in chunks:
        chunked_doc.append({'text': chunk, 'embedding': []})
    return chunked_doc
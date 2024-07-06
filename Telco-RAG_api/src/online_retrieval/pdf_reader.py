from src.online_retrieval.google_page import fetch_all_snippets
import os
import asyncio
import string
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import difflib
import httpx
from bs4 import BeautifulSoup
from urllib.parse import quote
from docx import Document
from PyPDF2 import PdfReader

# Ensure required NLTK resources are downloaded
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "nltk"])
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

def clean_snippet(snippet):
    return snippet.strip(string.punctuation)

def extract_centered_chunk(text, input_sentence, max_tokens=100):
    words = word_tokenize(text)
    sentence_words = word_tokenize(input_sentence)
    sentence_len = len(sentence_words)
    for i in range(len(words) - sentence_len + 1):
        if words[i:i + sentence_len] == sentence_words:
            start = max(0, i - (max_tokens - sentence_len) // 2)
            end = min(len(words), start + max_tokens)
            return ' '.join(words[start:end])
    return None

def find_most_similar_paragraph(snippet, text):
    # Check for an exact match
    exact_match_chunk = extract_centered_chunk(text, snippet)
    if exact_match_chunk:
        return exact_match_chunk

    paragraphs = text.split('\n\n')
    best_match = None
    best_ratio = 0.0

    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        for sentence in sentences:
            s = difflib.SequenceMatcher(None, snippet, sentence)
            ratio = s.ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = sentence

    # Ensure we always return a paragraph within the 100-token limit
    if best_match:
        return extract_centered_chunk(text, best_match, max_tokens=200)
    else:
        return extract_centered_chunk(text, paragraphs[0], max_tokens=200) if paragraphs else extract_centered_chunk(text, text, max_tokens=200)

async def fetch_snippets_and_search(query, question,model_name , validator, UI):
    engines = ["google"]
    # , f"{query} site:3gpp.org", f"{query} filetype:pdf"
    snippets_full = await fetch_all_snippets([query], question, model_name, engines, validator, UI)
    # print(f'We retrieved {len(snippets_full[query])} snippets')
    # print('-'*100)

    snippets_list = [snippet for snippet_list in snippets_full.values() for snippet in snippet_list]
    top_paragraphs = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda snippet: find_most_similar_paragraph(snippet['snippet'], snippet['text']), snippets_list))

    for i, snippet in enumerate(snippets_list):
        para = results[i]
        # print(f"Snippet length: {len(snippet['snippet'])}, Text length: {len(snippet['text'])}")
        top_paragraphs.append(f"Retrieval {i+11}:\n\nSNIPPET: {snippet['snippet']}\n\nEXTENDED VERSION: {para}\nThis retrieval is performed from the document {snippet['link']}\n")
    return top_paragraphs
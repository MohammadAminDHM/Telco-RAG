import logging
import numpy as np
import traceback
from src.LLMs.LLM import embedding

def get_embeddings_OpenAILarge_byapi(text):
    
    response = embedding(text)
    return response.data[0].embedding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_embeddings(series_docs):
    """Add embeddings to each chunk of documents from pre-saved NumPy files."""
    for doc_key, doc_chunks in series_docs.items():
        try:
            embeddings = np.load(f'3GPP-Release18\Embeddings\Embeddings{doc_key}.npy')
        except FileNotFoundError:
            logging.error(f"Embedding file for {doc_key} not found.")
            continue
        except Exception as e:
            logging.error(f"Failed to load embeddings for {doc_key}: {e}")
            continue
        
        text_list=[]
        for chunk in doc_chunks:
            for single_chunk in chunk:
                text_list.append(single_chunk['text'])
        dex ={}
        for i in range(len(text_list)):
            dex[text_list[i]] = embeddings[i]
        updated_chunks = []
        for chunk in doc_chunks:
            for idx, single_chunk in enumerate(chunk):
                try:
                    chunk[idx]['embedding'] = dex[chunk[idx]['text']]
                    updated_chunks.append(chunk[idx])
                except IndexError:
                    logging.warning(f"Embedding index {idx} out of range for {doc_key}.")
                except Exception as e:
                    logging.error(f"Error processing chunk {idx} for {doc_key}: {e}")
        
        series_docs[doc_key] = updated_chunks
    return series_docs


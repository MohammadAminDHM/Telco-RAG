import numpy as np
import os
import traceback
from src.LLMs.LLM import embedding
import logging

def search_faiss_index(faiss_index, query_embedding, k=5):
    if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a 1D numpy array")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    query_embedding_reshaped = query_embedding.reshape(1, -1)

    #search
    D, I = faiss_index.search(query_embedding_reshaped, k)

    return I, D

def get_query_embedding_OpenAILarge(query_text, context=None):
    try:
        if context is not None:
            query_text = f'{query_text}\n' + "\n".join(context)

        response = embedding(query_text)
        query_embedding =  response.data[0].embedding  
        return np.array(query_embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error occurred in get_query_embedding_OpenAILarge: {e}")
        traceback.print_exc()  
             
def find_nearest_neighbors_faiss(query_text, faiss_index, data_mapping, k, source_mapping, embedding_mapping,  context=None):
    
    try:
        query_embedding = get_query_embedding_OpenAILarge(query_text, context)

        I, D = search_faiss_index(faiss_index, query_embedding, k)

        nearest_neighbors = []
        for index in I[0]:  
            if index < len(data_mapping):  
                data = data_mapping.get(index, "Data not found")
                source = source_mapping.get(index, "Source not found")
                embedding = embedding_mapping.get(index, "Data not found")
                nearest_neighbors.append((index, data, source, embedding))
        return nearest_neighbors
    except Exception as e:
        print(f"Error in find_nearest_neighbors_faiss: {str(e)}")
        traceback.print_exc()
        return []
 
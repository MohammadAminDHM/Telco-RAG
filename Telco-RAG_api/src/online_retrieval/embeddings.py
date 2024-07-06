from openai import OpenAI


def get_embeddings_OpenAILarge_byapi(chunked_doc):
    text_list = []
    for chunk in chunked_doc:
        text_list.append(chunk["text"])
    
    client = OpenAI()
    response = client.embeddings.create(
                input=text_list,
                model="text-embedding-3-large",
                dimensions=1024,
            )
    embeddings = []
    for i in range(len(response.data)):
        embeddings.append(response.data[i].embedding)
    
    dex = dict()
    for i in range(len(embeddings)):
        dex[text_list[i]] = embeddings[i]
    
    for chunk in chunked_doc:
        chunk['embedding'] = dex[chunk['text']]
        chunk['source'] = 'Online'

    return chunked_doc
import os
import traceback
from src.query import Query
from api.generate import generate
from api.LLM import submit_prompt_flex_UI
import sys
import git
import asyncio
import time
from api.utils import update_secrets_file

folder_url = "https://huggingface.co/datasets/netop/Embeddings3GPP-R18"
clone_directory = "./3GPP-Release18"

if not os.path.exists(clone_directory):
    git.Repo.clone_from(folder_url, clone_directory)
    print("Folder cloned successfully!")
else:
    print("Folder already exists. Skipping cloning.")

async def TelcoRAG(query, model_name='gpt-3.5', api_key= None):
    if api_key == None:
        try:
            from api.settings.config import get_settings
            settings = get_settings()
            api_key = settings.openai__api_key
        except:
            sys.exit("You do not have an OpenAI api key stored.")
    try:
        update_secrets_file(model_name, api_key)
        start =  time.time()
        question = Query(query, [])

        query = question.question
        conciseprompt=f"""Rephrase the question to be clear and concise:
        
        {question.question}"""

       
        concisequery = submit_prompt_flex_UI(conciseprompt, model=model_name).rstrip('"')

        question.query = concisequery
        # question.question = concisequery

        question.def_TA_question()
        print()
        print('#'*50)
        print(concisequery)
        print('#'*50)
        print()

        try:
            loop = asyncio.get_event_loop()
            context_3gpp_future = loop.run_in_executor(None, question.get_3GPP_context, 10, model_name, False, True)
            online_info = await question.get_online_context_UI(model_name=model_name)

            await context_3gpp_future
        except:
            online_info = await question.get_online_context_UI(model_name=model_name)


        for online_parag in online_info:
            question.context.append(online_parag)
        
        response, context, _ = generate(question, model_name)
        end=time.time()
        print(f'Generation of this response took {end-start} seconds')
        return response, context, query
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())

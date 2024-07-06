import os
import traceback
from src.query import Query
from src.generate import generate, check_question
from src.LLMs.LLM import submit_prompt_flex
import traceback
import git
import asyncio
import time

folder_url = "https://huggingface.co/datasets/netop/Embeddings3GPP-R18"
clone_directory = "./3GPP-Release18"

if not os.path.exists(clone_directory):
    git.Repo.clone_from(folder_url, clone_directory)
    print("Folder cloned successfully!")
else:
    print("Folder already exists. Skipping cloning.")
    

async def TelcoRAG(query, answer=None, options=None, model_name='gpt-3.5'):
    try:
        start = time.time()
        question = Query(query, [])
        conciseprompt = f"Rephrase the question to be clear and concise:\n\n{question.question}"
        concisequery = submit_prompt_flex(conciseprompt, model=model_name).rstrip('"')
        question.query = concisequery
        question.def_TA_question()
        print('#'*50, concisequery, '#'*50, sep='\n')

        loop = asyncio.get_event_loop()
        context_3gpp_future = loop.run_in_executor(None, question.get_3GPP_context, 10, model_name, False, False)
        online_info = await question.get_online_context(model_name=model_name, validator_flag=False)
        await context_3gpp_future

        for online_parag in online_info:
            question.context.append(online_parag)
        
        print(answer)
        if answer is not None:
            response, context, _ = check_question(question, answer, options, model_name=model_name)
            print(context)
        else:
            response, context, _ = generate(question, model_name)
        
        end = time.time()
        print(f'Generation of this response took {end - start} seconds')
        return response, question.context

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    question =  {
        "question": "In supporting an MA PDU Session, what does Rel-17 enable in terms of 3GPP access over EPC? [3GPP Release 17]",
        "options" : { 
        "option 1": "Direct connection of 3GPP access to 5GC",
        "option 2": "Establishment of user-plane resources over EPC",
        "option 3": "Use of NG-RAN access for all user-plane traffic",
        "option 4": "Exclusive use of a non-3GPP access for user-plane traffic"
        },
        "answer": "option 2: Establishment of user-plane resources over EPC",
        "explanation": "Rel-17 enables the establishment of user-plane resources over EPC for 3GPP access in supporting an MA PDU Session, allowing for simultaneous traffic over EPC and non-3GPP access.",
        "category": "Standards overview"
    }
    # Example using an MCQ
    response, context = asyncio.run(TelcoRAG(question['question'], question['answer'], question['options'], model_name='gpt-3.5' ))
    print(response, '\n')
    # Example using an open-end question           
    response, context = asyncio.run(TelcoRAG(question['question'], model_name='gpt-3.5' ))
    print(response, '\n')
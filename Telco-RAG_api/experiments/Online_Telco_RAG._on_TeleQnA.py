import os
import traceback
import random
import json
import asyncio
import git
import openai
import numpy as np
import time
from src.embeddings import get_embeddings
from src.query import Query
from src.generate import generate, check_question
from src.LLMs.LLM import submit_prompt_flex
from src.input import get_documents
from src.chunking import chunk_doc

folder_url = "https://huggingface.co/datasets/netop/Embeddings3GPP-R18"
clone_directory = "./3GPP-Release18"

if not os.path.exists(clone_directory):
    git.Repo.clone_from(folder_url, clone_directory)
    print("Folder cloned successfully!")
else:
    print("Folder already exists. Skipping cloning.")

def choose_random_questions(data, num_q=500, filename='questions.json'):
    result_list = []
    result_dict = {}
    selected_questions = set()
    
    while len(result_list) < num_q:
        random_question = random.choice(list(data.values()))
        if '3GPP' in random_question['question'] and random_question['question'] not in selected_questions: 
            options = [f"{elem}: {random_question[elem]}" for elem in random_question.keys() if 'option' in elem]
            question_tuple = (random_question['question'], random_question['answer'], options)
            result_list.append(question_tuple)
            result_dict[random_question['question']] = question_tuple
            selected_questions.add(random_question['question'])
    
    with open(filename, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
    
    return result_list

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
        context_3gpp_future = loop.run_in_executor(None, question.get_3GPP_context, 10, model_name)
        online_info = await question.get_online_context(model_name=model_name)
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
        return response, question.context, query

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    with open("TeleQnA.json", "r") as file:
        questions = json.load(file)

    Q3GPP_list = [{'question': q['question'], 'answer': q['answer'], 'options': [f"{elem}: {q[elem]}" for elem in q.keys() if 'option' in elem]} for q in questions.values()]

    correctness = []
    start_count = time.time()

    try:
        with open("benchmarkGPT-TeleQnA-NOvalidator.json", "r") as file:
            Qtested = json.load(file)
    except:
        Qtested = []

    Q_alredy_tested = [q['question']['question'] for q in Qtested]
    print(len(Q_alredy_tested))

    for question in Qtested:
        correctness.append(question['response'])

    Q3GPP_list_untested = [q for q in Q3GPP_list if q['question'] not in Q_alredy_tested]
    question_count = len(Qtested)

    for question in Q3GPP_list_untested:
        try:
            print(f'Total number of questions to be tested: {len(Q3GPP_list_untested)}')
            response, context = asyncio.run(TelcoRAG(question['question'], question['answer'], question['options'], model_name='gpt-3.5'))
            question_count += 1

            if os.path.exists("benchmarkGPT-TeleQnA-NOvalidator.json"):
                with open("benchmarkGPT-TeleQnA-NOvalidator.json", "r") as file:
                    try:
                        log_data = json.load(file)
                    except json.JSONDecodeError:
                        log_data = []
            else:
                log_data = []

            log_data.append({
                "question": question,
                "context": context,
                "answer": question['answer'],
                "options": question['options'],
                "response": response
            })

            with open("benchmarkGPT-TeleQnA-NOvalidator.json", "w") as file:
                json.dump(log_data, file, indent=4)

            correctness.append(response)
            end_count = time.time()
            print('#'*100)
            print(f"Answer to the question is {response}")
            print(f"Average accuracy is: {np.mean(correctness)}")
            print(f"Total questions tested so far: {question_count}")
            print(f"Average time per question is {(end_count - start_count) / question_count}")
            print('#'*100)
        except:
            pass

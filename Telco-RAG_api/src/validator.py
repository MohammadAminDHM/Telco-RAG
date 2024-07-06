
import concurrent.futures
import time
from src.LLMs.LLM import submit_prompt_flex
from api.LLM import submit_prompt_flex_UI

def LLM_validator_snippet(paragraph, query, model_name):
    input_prompt = f"""You are an telecom expert and you are given the following snippet that describes the content of a document.

    Query: {query}
    Paragraph: {paragraph}
    
    Tell if it would be possible that the document contains relevant information, possibly helpful to respond the given query in a precise and complete way!
    1. YES
    2. NO"""

    response = submit_prompt_flex(input_prompt, model=model_name)

    return response

def LLM_validator_snippet_UI(paragraph, query, model_name):
    input_prompt = f"""You are an telecom expert and you are given the following snippet that describes the content of a document.

    Query: {query}
    Paragraph: {paragraph}
    
    Tell if it would be possible that the document contains relevant information, possibly helpful to respond the given query in a precise and complete way!
    1. YES
    2. NO"""

    response = submit_prompt_flex_UI(input_prompt, model=model_name)

    return response

def LLM_validator_RAG(paragraph, query, model_name):
    input_prompt = f"""Tell if the following paragraph contains relevant information, possibly helpful to respond the given query in a precise and complete way!
    Query: {query}
    Paragraph: {paragraph}
    
    Tell if the following paragraph contains relevant information, possibly helpful to respond the given query in a precise and complete way!
    1. YES
    2. NO"""

    response = submit_prompt_flex(input_prompt, model=model_name)

    return response

def LLM_validator_RAG_UI(paragraph, query, model_name):
    input_prompt = f"""Tell if the following paragraph contains relevant information, possibly helpful to respond the given query in a precise and complete way!
    Query: {query}
    Paragraph: {paragraph}
    
    Tell if the following paragraph contains relevant information, possibly helpful to respond the given query in a precise and complete way!
    1. YES
    2. NO"""

    response = submit_prompt_flex_UI(input_prompt, model=model_name)

    return response

def validator_online(question, context_list, model_name='gpt-3.5', UI_flag=True):
    start_time_overall = time.time()
    if UI_flag is False:
        def check_context_item(context_item):
            start_time = time.time()
            is_valid = 'no' not in LLM_validator_snippet(context_item, question, model_name=model_name).lower()
            end_time = time.time()
            duration = end_time - start_time
            return (context_item if is_valid else None, duration)
    else:
        def check_context_item(context_item):
            start_time = time.time()
            is_valid = 'no' not in LLM_validator_snippet_UI(context_item, question, model_name=model_name).lower()
            end_time = time.time()
            duration = end_time - start_time
            return (context_item if is_valid else None, duration)
        
    final_list = []
    durations = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for context_item, duration in executor.map(check_context_item, context_list):
            if context_item is not None:
                final_list.append(context_item)
            durations.append(duration)
    
    # for duration in durations:
    #     print(f'Validation time: {duration:.4f} seconds')
    
    # print(f'Reduced from {len(context_list)} to {len(final_list)}-online')
    end_time_overall = time.time()
    duration = end_time_overall - start_time_overall
    print(f'Overall check lasted for {duration:.4f} seconds')
    
    return final_list

import concurrent.futures
import time

def validator_RAG(question, context_list, model_name='gpt-3.5', k=10, UI_flag=True):
    start_time_overall = time.time()
    
    if UI_flag is False:
        def check_context_item(context_item):
            start_time = time.time()
            is_valid = 'no' not in LLM_validator_RAG(context_item, question, model_name=model_name).lower()
            end_time = time.time()
            duration = end_time - start_time
            return (context_item if is_valid else None, duration)
    else:
        def check_context_item(context_item):
            start_time = time.time()
            is_valid = 'no' not in LLM_validator_RAG_UI(context_item, question, model_name=model_name).lower()
            end_time = time.time()
            duration = end_time - start_time
            return (context_item if is_valid else None, duration)

    final_list = []
    durations = []

    batch_list = []
    for i in range(2):
        batch_list.append(context_list[i*k:(i+1)*k])

    for batch in batch_list:
        if len(final_list) < 5: 
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for context_item, duration in executor.map(check_context_item, batch):
                    if context_item is not None:
                        final_list.append(context_item)
                    durations.append(duration)
    
    # for duration in durations:
    #     print(f'Validation time: {duration:.4f} seconds')
    
    # print(f'Reduced from {len(context_list)} to {len(final_list)}-RAG')
    end_time_overall = time.time()
    duration = end_time_overall - start_time_overall
    print(f'Overall check lasted for {duration:.4f} seconds')
    
    return final_list


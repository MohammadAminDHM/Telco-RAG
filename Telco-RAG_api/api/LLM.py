import openai
import tiktoken

import asyncio

import anthropic # type: ignore
from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient

from together import AsyncTogether, Together


import time

from api.settings.config import get_settings
from groq import Groq, AsyncGroq

import platform
if platform.system()=='Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
settings = get_settings()
rate_limit = settings.rate_limit 

# API keys
openai.api_key = settings.openai_api_key
any_api_key = settings.any_api_key
mistral_api = settings.mistral_api
anthropic_api = settings.anthropic_api
cohere_api = settings.cohere_api
pplx_api = settings.pplx_api
together_api = settings.together_api

groq_api = ""

# Models config
models = [
    "gpt-3.5",
    "gpt-4",
    "gpt-4o",
    'mixtral',
    'mistral-small',
    'mistral-medium',
    "code-llama",
    "command-R+",
    'pplx',
    'mixtral-8x22',
    "mixtral-groq",
    'llama-3',
    'llama-3-any',
    'llama-3-8B',
    'wizard'
]

models_fullnames = {
    "gpt-3.5": "gpt-3.5-turbo-0125",
    "gpt-4": "gpt-4-turbo-2024-04-09",
    "gpt-4o": "gpt-4o-2024-05-13",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral-small-old": 'open-mixtral-8x7b',
    "mistral-small": 'mistral-small-latest',
    "mistral-medium": 'mistral-medium-latest',
    "mistral-large": 'mistral-large-latest',
    "code-llama": "codellama/CodeLlama-70b-Instruct-hf",
    "claude-small": "claude-3-haiku-20240307",
    "claude-medium": "claude-3-sonnet-20240229",
    "claude-large": "claude-3-opus-20240229",
    "command-R+": "command-r-plus",
    "pplx": "llama-3-sonar-large-32k-online",
    'mixtral-8x22': "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mixtral-groq": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    'llama-3': 'meta-llama/Llama-3-70b-chat-hf',
    'llama-3-any': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama-3-8B' : 'meta-llama/Llama-3-8b-chat-hf',
    'wizard': "microsoft/WizardLM-2-8x22B"
}

models_endpoints = {
    "gpt-3.5": "openai",
    "gpt-4": "openai",
    "gpt-4o": "openai",
    "mixtral": "anyscale",
    "mistral-small-old": 'mistral',
    "mistral-small": 'mistral',
    "mistral-medium": 'mistral',
    "mistral-large": 'mistral',
    "code-llama": "anyscale",
    "claude-small": "anthropic",
    "claude-medium": "anthropic",
    "claude-large": "anthropic",
    "command-R+": "cohere",
    "pplx": "perplexity",
    'mixtral-8x22': 'anyscale',
    "mixtral-groq": 'groq',
    'llama-3': 'together',
    'llama-3-any': 'anyscale',
    'llama-3-8B': 'anyscale',
    'wizard': 'together'
}

token_prices = {
    "gpt-3.5": 0.0015 / 1000,
    "gpt-4": 0.06 / 1000,
    "mixtral": 0.5 / 1000000,
    "mixtral-groq": 0.5 / 1000000,
    "mistral-medium": 5 / 1000000,
}


class RateLimiter:
    def __init__(self, calls_per_second=0.25):
        self.calls_per_second = rate_limit
        self.semaphore = asyncio.Semaphore(calls_per_second)
        self.next_call_time = time.time()

    async def wait_for_rate_limit(self):
        async with self.semaphore:
            now = time.time()
            sleep_time = self.next_call_time - now
            self.next_call_time = max(self.next_call_time + 1 / self.calls_per_second, now)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            self.next_call_time = max(self.next_call_time + 1 / self.calls_per_second, now)


def submit_prompt_flex_UI(prompt, model="gpt-3.5", output_json=False):
    if model in models_fullnames:
        model_fullname = models_fullnames[model]
        endpoint = models_endpoints[model]
    else:
        endpoint = ""
        
    if endpoint == "anyscale":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.OpenAI(
            base_url = "https://api.endpoints.anyscale.com/v1",
            api_key=any_api_key,
        )
        generate = client.chat.completions.create
    elif endpoint == "perplexity":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.OpenAI(
            base_url = "https://api.perplexity.ai",
            api_key=any_api_key,
        )
        generate = client.chat.completions.create
    elif endpoint == "groq":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = Groq(
            api_key=groq_api,
        )
        generate = client.chat.completions.create
    elif endpoint == "together":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")     
        client = Together(api_key=together_api)  
        generate = client.chat.completions.create 
    elif endpoint == "openai":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.OpenAI(
            api_key=openai.api_key,
        )
        generate = client.chat.completions.create
    elif endpoint == "mistral":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = MistralClient(api_key=mistral_api)
        generate = client.chat
    elif endpoint == "anthropic":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = anthropic.Anthropic(
            api_key=anthropic_api,
        )
        def generate(**kwargs):
            return client.messages.create(
                max_tokens=4000,
                **kwargs
            )
    else:
        model_fullname = model
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.OpenAI(
            base_url = "https://api.endpoints.anyscale.com/v1",
            api_key=any_api_key,
        )
        generate = client.chat.completions.create        

    if output_json:
        generated_output = generate(
          model=model_fullname,
          response_format={"type":"json_object"},
          messages=[
              {"role": "user", "content": prompt}, 
            ]
        )
        if endpoint != "anthropic":
            output = generated_output.choices[0].message.content
        else:
            output = generated_output.content[0].text
            
        output = output.replace('"\n', '",\n')
        output = output[:output.rfind("}")+1]
        
    else:
        generated_output = generate(
          model=model_fullname,
          messages=[
              {"role": "user", "content": prompt}, 
            ]
        )
        if endpoint != "anthropic":
            output = generated_output.choices[0].message.content
        else:
            output = generated_output.content[0].text
    
    return output


async def a_submit_prompt_flex_UI(prompt, model="gpt-3.5", output_json=False):
    if model in models_fullnames:
        model_fullname = models_fullnames[model]
        endpoint = models_endpoints[model]
    else:
        endpoint = ""
        
    if endpoint == "anyscale":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.AsyncOpenAI(
            base_url = "https://api.endpoints.anyscale.com/v1",
            api_key=any_api_key,
        )
        generate = client.chat.completions.create
    elif endpoint == "perplexity":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.AsyncOpenAI(
            base_url = "https://api.perplexity.ai",
            api_key=any_api_key,
        )
        generate = client.chat.completions.create
    elif endpoint == "groq":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = AsyncGroq(
            api_key=groq_api,
        )
        generate = client.chat.completions.create
    elif endpoint == "together":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")     
        client = AsyncTogether(api_key=together_api)  
        generate = client.chat.completions.create 
    elif endpoint == "openai":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.AsyncOpenAI(
            api_key=openai.api_key,
        )
        generate = client.chat.completions.create
    elif endpoint == "mistral":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = MistralAsyncClient(api_key=mistral_api)
        generate = client.chat
    elif endpoint == "anthropic":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = anthropic.AsyncAnthropic(
            api_key=anthropic_api,
        )
        async def generate(**kwargs):
            return await client.messages.create(
                max_tokens=4000,
                **kwargs
            )
    else:
        model_fullname = model
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.AsyncOpenAI(
            base_url = "https://api.endpoints.anyscale.com/v1",
            api_key=any_api_key,
        )
        generate = client.chat.completions.create        

    if output_json:
        generated_output = await generate(
          model=model_fullname,
          response_format={"type":"json_object"},
          messages=[
              {"role": "user", "content": prompt}, 
            ]
        )
        if endpoint != "anthropic":
            output = generated_output.choices[0].message.content
        else:
            output = generated_output.content[0].text
            
        output = output.replace('"\n', '",\n')
        output = output[:output.rfind("}")+1]
        
    else:
        generated_output = await generate(
          model=model_fullname,
          messages=[
              {"role": "user", "content": prompt}, 
            ]
        )
        if endpoint != "anthropic":
            output = generated_output.choices[0].message.content
        else:
            output = generated_output.content[0].text
    
    return output

def embedding(input, dimension=1024):
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.embeddings.create(
                    input=input,
                    model="text-embedding-3-large",
                    dimensions=dimension,
                )
    return response
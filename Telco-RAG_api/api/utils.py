import toml
import os
from copy import deepcopy

def update_secrets_file(model, api_key):
    file_path = r'.\\api\\settings\\.secrets.toml'

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            config = toml.load(file)
    else:
        config = {
            'openai_api_key': "",
            'any_api_key' : "",
            'mistral_api' : "",
            'anthropic_api' : "",
            'cohere_api' : "",
            'google_search_api' : "",
            'pplx_api' : "",
            'together_api' : "",
            'rate_limit' : 2
            }
    
    secrets =  deepcopy(config)
    secrets['openai_api_key'] = api_key
    config.update(secrets)
    with open(file_path, 'w') as file:
        toml.dump(config, file)

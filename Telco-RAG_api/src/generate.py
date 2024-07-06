import re
import traceback
from src.LLMs.LLM import submit_prompt_flex
import logging

def generate(question, model_name):
    """Generate a response using the GPT model given a structured question object."""
    try:
        # Constructing the content context from the question object
        content = '\n'.join(question.context)
        prompt = f"""
Please answer the following question:
{question.query}

Considering the following context:
{content}

Please answer the following question, add between paranthesis the retrieval(e.g. Retrieval 3) that you used for each eleement of your reasoning:
{question.question}
        """
        print(prompt)
        logging.info("Generated system prompt for OpenAI completion.")
        
        predicted_answers_str = submit_prompt_flex(prompt, model=model_name)
        logging.info("Model response generated successfully.")

        context = f"The retrieved context provided to the LLM is:\n{content}"
        return predicted_answers_str, context, question.question

    except Exception as e:
        # Logging the error and returning a failure indicator
        logging.error(f"An error occurred: {e}")
        return None, None, None
    
def find_option_number(text):
    """
    Finds all the occurrences of numbers preceded by the word 'option' in a given text.

    Parameters:
    - text: The text to search for 'option' followed by numbers.

    Returns:
    - A list of strings, each representing a number found after 'option'. The numbers are returned as strings.
    - If no matches are found, an empty list is returned.
    """
    try:
        text =  text.lower()
        # Define a regular expression pattern to find 'option' followed by non-digit characters (\D*), 
        # and then one or more digits (\d+)
        pattern = r'option\D*(\d+)'
        # Find all matches of the pattern in the text
        matches = re.findall(pattern, text)
        return matches  # Return the list of found numbers as strings
    except Exception as e:
        print(f"An error occurred while trying to find option numbers in the text: {e}")
        return []


def check_question(question, answer, options, model_name='gpt-3.5'):
    """
    This function checks if the answer provided for a non-JSON formatted question is correct. 
    It dynamically selects the model based on the model_name provided and constructs a prompt 
    for the AI model to generate an answer. It then compares the generated answer with the provided 
    answer to determine correctness.

    Parameters:
    - question: A dictionary containing the question, options, and context.
    - model_name: Optional; specifies the model to use. Defaults to 'mistralai/Mixtral-8x7B-Instruct-v0.1' 
    if not provided or if the default model is indicated.

    Returns:
    - A tuple containing the updated question dictionary and a boolean indicating correctness.
    """
    try:
        # Extracting options from the question dictionary.
        options_text = '\n'.join(options)

        content = '\n'.join(question.context)
    
        syst_prompt = f"""
        Please provide the answers to the following multiple choice question.
        {question.query}
        
        Considering the following context:
        {content}
        
        Please provide the answers to the following multiple choice question.
        {question.question}
        
        Options:
        Write only the option number corresponding to the correct answer:\n{options_text}
        
        Answer format should be: Answer option <option_id>
        """
        print(syst_prompt)
        # Generating the model's response based on the constructed prompt.
        predicted_answers_str = submit_prompt_flex(syst_prompt, model=model_name)
        predicted_answers_str = predicted_answers_str.replace('"\n', '",\n')
        print(predicted_answers_str)
        print(answer)

        # Finding and comparing the predicted answer to the actual answer.
        answer_id = find_option_number(predicted_answers_str)

        if find_option_number(answer) == answer_id:
            print("Correct\n")
            return  True, f"Option {answer_id}", syst_prompt
        else:
            print("Wrong\n")
            return  False, f"Option {answer_id}", syst_prompt
    except Exception as e:
        # Error handling to catch and report errors more effectively.
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        return None, False
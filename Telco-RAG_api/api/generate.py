
import logging
from api.LLM import submit_prompt_flex_UI


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
        
        predicted_answers_str = submit_prompt_flex_UI(prompt, model=model_name)
        logging.info("Model response generated successfully.")

        context = f"The retrieved context provided to the LLM is:\n{content}"
        return predicted_answers_str, context, question.question

    except Exception as e:
        # Logging the error and returning a failure indicator
        logging.error(f"An error occurred: {e}")
        return None, None, None
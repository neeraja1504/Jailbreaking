import openai 
import json
import time
# from datasets import Dataset
from openai import AzureOpenAI
# from json_repair import json_repair
from tqdm import tqdm
client = AzureOpenAI(
    azure_endpoint="ENDPOINT",
    api_key="API_KEY",
    api_version="API_VERSION"
)


def query_openai_model(instruction: str, model_name="gpt4o"):
    message_text = [{"role": "system", "content": "You are a helpful agent"},
                    {"role": "user", "content": instruction}]
    completion = client.chat.completions.create(model=model_name,
                                                messages=message_text,
                                                temperature = 0,
                                                max_tokens = 3
                                                )

    response = completion.choices[0].message.content
    # output = json_repair.repair_json(response, return_objects=True)
    return response

def gpt_predict(window):
    """
    Function to prompt GPT with a sliding window of tokens and get the next word prediction.
    
    Parameters:
    window: list of tokens (the current sliding window)
    
    Returns:
    predicted_next_word: The word predicted by GPT to come next in the paragraph.
    """
    # Join the tokens in the sliding window to form a string for GPT's input
    prompt_text = f"Based on your memory complete the following text, only give the continuation nothing else: {' '.join(window)}"
    print("prompt_text is",prompt_text)
    
    response = query_openai_model(prompt_text)

    # Extract the predicted word from the GPT response
    # predicted_next_word = response['choices'][0]['text'].strip()
    predicted_next_word = response.split(" ")[0].strip()
    print(predicted_next_word)

    return predicted_next_word


def compute_accuracy_with_gpt(paragraph):
    """
    Computes the accuracy using a sliding window where the window size increases incrementally.
    The true label is the next word in the paragraph.
    
    Parameters:
    paragraph: list of tokens (words)
       - This is the context, where each token is a word or token in the paragraph.
    
    Returns:
    accuracy: The proportion of correctly predicted next words
    """
    correct_predictions = 0
    total_windows = len(paragraph) - 1  # Number of windows is paragraph length minus 1 for next word

    # Iterate over the paragraph with variable window size
    for window_size in range(1, len(paragraph)):
        # Extract the window of tokens from the paragraph (up to window_size tokens)
        window = paragraph[:window_size]

        # The true label is the next word after the window
        true_label = paragraph[window_size]  # This is the next word in the paragraph

        # Get the predicted next word using GPT
        predicted_word = gpt_predict(window)

        # Check if the predicted word matches the true label (next word in the paragraph)
        if predicted_word == true_label:
            correct_predictions += 1

    # Calculate accuracy as the proportion of correct predictions
    accuracy = correct_predictions / total_windows
    return accuracy

paragraph = '''I wandered lonely as a cloud
              That floats on high o'er vales and hills,
              When all at once I saw a crowd,
              '''
print(paragraph)
paragraph = paragraph.split()
accuracy = compute_accuracy_with_gpt(paragraph)
print(f"Accuracy: {accuracy}")

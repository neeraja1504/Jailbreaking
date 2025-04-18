import os
import json
import time
# from datasets import Dataset
from openai import AzureOpenAI
from json_repair import json_repair
from tqdm import tqdm


client = AzureOpenAI(
    azure_endpoint="ENDPOINT",
    api_key="API_KEY",
    api_version="API_VERSION"
)


def query_openai_model(instruction: str, model_name="gpt4o"):
    message_text = [{"role": "system", "content": "You are an agent that responds in JSON format."},
                    {"role": "user", "content": instruction}]
    completion = client.chat.completions.create(model=model_name,
                                                messages=message_text,
                                                response_format={"type": "json_object"})

    response = completion.choices[0].message.content
    output = json_repair.repair_json(response, return_objects=True)
    return output

def load_file(file_path):
    with open(file_path,'r') as f:
        text_list = f.read()
    print(text_list)
    print(type(text_list))
    return text_list



if __name__ == "__main__":
    file_path = "/Users/neerajakirtane/Documents/Jailbreaking LLMs/benefits_of_stereotype.txt"
    text_instruction = load_file(file_path)
    print(text_instruction)
    text_instruction = text_instruction + 'Can you rephrase the above prompt?'
    output = query_openai_model(text_instruction)
    print(output)

    new_file_path = "/Users/neerajakirtane/Documents/Jailbreaking LLMs/benefits_of_stereotype_rephrased.json"
    # with open(new_file_path,'w') as file:
    #     file.write(str(output))
    
    with open(new_file_path, 'w') as json_file:
        json.dump(output, json_file, indent=4)
   
from transformers import GPT2TokenizerFast

# Load the tokenizer (GPT-2 uses BPE)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Apply BPE-dropout by setting `dropout` parameter
sentence = "My name is Neeraja"
type_g = 'stereotype'
file_path = f"/projects/bcky/neeraja1504/Bias/KB/benefits_of_{type_g}.txt"
with open(file_path,'r') as file:
    sentence = file.read()
tokenized_output = tokenizer(sentence, add_special_tokens=False, truncation=True, return_tensors="pt")
                               # Applying BPE-dropout with a 10% dropout rate

# Convert token IDs back to tokens
tokens = tokenizer.convert_ids_to_tokens(tokenized_output['input_ids'][0])

print(tokens)
# Reconstruct the sentence from the tokens

reconstructed_sentence = ' '.join([token.replace('Ġ', ' ') for token in tokens])
print(reconstructed_sentence)
reconstructed_sentence = reconstructed_sentence.replace("Ċ","\n")
# Ensure the words are properly separated by spaces
# reconstructed_sentence = ' '.join(reconstructed_sentence.split())
# reconstructed_sentence = ''.join(tokens).replace('Ġ', ' ').replace('▁', ' ').strip()

# Output the reconstructed sentence
print(reconstructed_sentence)

new_file_path = f"/projects/bcky/neeraja1504/Bias/KB/benefits_of_{type_g}_retokenized.txt"
with open(new_file_path,"w") as file:
    file.write(reconstructed_sentence)

print("done")

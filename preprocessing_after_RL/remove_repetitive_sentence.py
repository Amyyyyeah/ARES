import os
import argparse
import json
import torch
from transformers import AutoTokenizer
from utils import check_eos
from tqdm import tqdm

def main(args):
    
    # Read JSON file
    with open(args.file_path, 'r') as file:
        data = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for row_idx in tqdm(range(len(data['preds']))):
        rationale = data['preds'][row_idx]
        decoded_sentences_list = []
        encoded_rationale = torch.tensor([tokenizer.encode(rationale)])
        sentences_seen = {}  # Dictionary to track seen sentences 
        start = 0
        for col_idx in range(len(encoded_rationale[0])):
            mask = check_eos(encoded_rationale, col_idx)

            if mask[0] == True:

                decoded_sentence = tokenizer.decode(encoded_rationale[0][start:col_idx+1], skip_special_tokens=True)
                if len(decoded_sentence) > 0 and decoded_sentence[0] == 'n':
                    decoded_sentence = decoded_sentence[1:]
                if decoded_sentence not in sentences_seen:
                    sentences_seen[decoded_sentence] = True
                    decoded_sentences_list.append(f'{decoded_sentence}')
                start = col_idx + 1

            
        if len(decoded_sentences_list) > 0:
            decoded_sentences_list[0] = decoded_sentences_list[0].replace("Solution: ", "").strip()
            decoded_sentences_list[0] = decoded_sentences_list[0].replace("Solution:", "").strip()

        concatenated_sentences = ' '.join(decoded_sentences_list)
        data['preds'][row_idx] = concatenated_sentences
    
    dir  = os.path.dirname(args.file_path)
    new_file_path = os.path.join(dir, "remove_repetitive_sentences.json")

    # Save the modified data to the new file path
    with open(new_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print('Done removing repetitive sentences. Output saved to:', new_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./test_train.json')
    parser.add_argument('--tokenizer', type=str, default='./models/mm-cot-base-rationale/')
    
    args = parser.parse_args()

    main(args)
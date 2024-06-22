import subprocess
import time
import json
import os
import shutil
import re
from transformers import AutoTokenizer

original_data = './scienceqa_data/ScienceQA/data/scienceqa/problems.json'

with open(original_data) as f:
    original_data = json.load(f)

processing_data = './RL_models/base_neutral0.5_k4_rlb8_cl0.2_rle10_lr2e-05_vlr1.0_g1.0_l0.95_fGPT4V_seed42_kl0.0001_ga8_dosampleTrue_advTrue_tk50_ref/0/remove_repetitive_sentences.json'

with open(processing_data) as f:
    processing_data = json.load(f)

def extract_number(dir_name):
    try:
        return int(dir_name)
    except ValueError:
        return float('inf')

directory = './preprocessing_after_RL/correction/0/p0' # Please create the folder with the PIDs of "train" under this directory
subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

sorted_subdirs = sorted(subdirs, key=extract_number)

subdirs_idx = 0
for key, value in original_data.items():
    if value['split'] == 'train':
        if key == sorted_subdirs[subdirs_idx]:
            file_path = os.path.join(directory,key)
            if processing_data['preds'][subdirs_idx]!='':
                contents = "{\n" + processing_data['preds'][subdirs_idx] + "\n}"
            else:
                contents = "{\n}"
            with open(os.path.join(file_path, 'rationale_from_mm_cot.txt'), 'w') as f:
                f.write(contents)
            subdirs_idx += 1

total_image_dir = './preprocessing_after_RL/images/scienceqa'
for key, value in original_data.items():
    #if value['split'] == 'train':
    if key in sorted_subdirs:
        dir_path = os.path.join(directory, key)
        each_image_folder = os.path.join(total_image_dir, key)
        with open(os.path.join(dir_path, 'rationale_from_mm_cot.txt'), 'r') as rationale_file:
            rationales_dict = rationale_file.read()
        answer_index=value['answer']
        answer_text=value['choices'][answer_index]
        if value['image']=='image.png':
            source_file = os.path.join(each_image_folder, 'image.png')
            destination_file = os.path.join(dir_path, 'image.png')
            shutil.copy(source_file, destination_file)
            input_text = f"""\
Your task involves reviewing a set that includes an Image, Options, Hint, Answer, and Rationales for a Question. \
Please follow below 7 rules.\n\
1. Preserve any correct original rationales based on the given answer by incorporating them into the final rationale without making any alterations.\n\
2. Preserve any original rationales that represent the starting point of thought.\n\
3. Correct any grammatical errors or incomplete rationales based on the given information without your knowledge.\n\
4. If there are incorrect rationales based on the given answer, please correct them without removing them based on the given information.\n\
5. Please take into account the content of the options, hint, and answer when doing this task.\n\
6. Fill the corrected rationales inside the {{}} in the final_rationale according to the given format below, without any additional explanation.\n\
7. Return only the entire set of Rationales within curly braces ({{}}) below with the filled one in the step 6.\n\n\
Question: {value['question']}\nOptions: {value['choices']}\nHint: {value['hint']}\nAnswer: {answer_text}\n\nRationales:\n{{\noriginal_rationale:{rationales_dict}, \nfinal_rationale:{{}}\n}}"""
        else:
            input_text = f"""\
Your task involves reviewing a set that includes Options, Hint, Answer, and Rationales for a Question. \
Please follow below 7 rules.\n\
1. Preserve any correct original rationales based on the given answer by incorporating them into the final rationale without making any alterations.\n\
2. Preserve any original rationales that represent the starting point of thought.\n\
3. Correct any grammatical errors or incomplete rationales based on the given information without your knowledge.\n\
4. If there are incorrect rationales based on the given answer, please correct them without removing them based on the given information.\n\
5. Please take into account the content of the options, hint, and answer when doing this task.\n\
6. Fill the corrected rationales inside the {{}} in the final_rationale according to the given format below, without any additional explanation.\n\
7. Return only the entire set of Rationales within curly braces ({{}}) below with the filled one in the step 6.\n\n\
Question: {value['question']}\nOptions: {value['choices']}\nHint: {value['hint']}\nAnswer: {answer_text}\n\nRationales:\n{{\noriginal_rationale:{rationales_dict}, \nfinal_rationale:{{}}\n}}"""
        final_input_path = os.path.join(dir_path, 'input_for_llm.txt')
        print(dir_path)
        with open(final_input_path, 'w') as file:
            file.write(input_text)





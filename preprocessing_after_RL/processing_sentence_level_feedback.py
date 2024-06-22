import time
import json
import os
import shutil
import re


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_actions', type=int, default=4)
    parser.add_argument('--problems', type=str, default="scienceqa")
    args = parser.parse_args()
    os.chdir('./')
    base_dir = f"./preprocessing_after_RL/{args.k_actions}"
    total_image_dir = f"./preprocessing_after_RL/images/{args.problems}"
    question_file = f"./{args.problems}_data/ScienceQA/data/{args.problems}/problems.json"
    
    mmcot_finished_signal_file = os.path.join(base_dir, "mmcot_finished_signal.txt")
    
    while True:
        if os.path.exists(mmcot_finished_signal_file):
            if os.path.isfile(mmcot_finished_signal_file):
                os.remove(mmcot_finished_signal_file)
            for i_action in os.listdir(base_dir):
                print(f"{i_action} folder start!")
                items_in_directory = os.listdir(i_action_folder_path)
                for item in items_in_directory:
                    item_path = os.path.join(i_action_folder_path, item)
                    if os.path.isfile(item_path) and item.endswith('.txt'):
                        try:
                            os.remove(item_path)
                            print(f"Deleted: {item_path}")
                        except Exception as e:
                            print(f"Error deleting {item_path}: {e}")
                '''
                ################## Load ##################
                '''
                with open(question_file) as f:
                    data = json.load(f)
                
                for key, value in data.items():
                    if value['split'] == 'train':
                        dir_path = os.path.join(i_action_folder_path, key)
                        each_image_folder = os.path.join(total_image_dir, key)
                        answer_index=value['answer']
                        answer_text=value['choices'][answer_index]
                        if not os.path.exists(each_image_folder):
                            os.makedirs(each_image_folder)
                        if os.path.exists(dir_path):
                            with open(os.path.join(dir_path, 'question.txt'), 'w') as f:
                                f.write(value['question'])
                            with open(os.path.join(dir_path, 'answer.txt'), 'w') as f:
                                f.write(answer_text)
                            with open(os.path.join(dir_path, 'choice.txt'), 'w') as f:
                                f.write(str(value['choices']))
                            with open(os.path.join(dir_path, 'hint.txt'), 'w') as f:
                                f.write(str(value['hint']))
                
                subfolders = [f.path for f in os.scandir(i_action_folder_path) if f.is_dir()]
                
                for subfolder in subfolders:
                    with open(os.path.join(subfolder, 'question.txt'), 'r') as question_file:
                        question = question_file.read().strip()
                    with open(os.path.join(subfolder, 'answer.txt'), 'r') as answer_file:
                        answer = answer_file.read().strip()
                    with open(os.path.join(subfolder, 'choice.txt'), 'r') as choice_file:
                        choice = choice_file.read().strip()
                    with open(os.path.join(subfolder, 'hint.txt'), 'r') as hint_file:
                        hint = hint_file.read().strip()
                    with open(os.path.join(subfolder, 'rationale_from_mm_cot.txt'), 'r') as rationale_file:
                        rationales_dict = rationale_file.read()
                    ck_folder_name = os.path.basename(subfolder)
                    ck_image_file =  os.path.join(total_image_dir, ck_folder_name)
                    print(ck_image_file)
                    if os.path.exists(os.path.join(ck_image_file, 'default_image.png')):
                        input_text = f"""There exists a set comprising Options, Hint, and Answer for a Question. The reasoning process used to deduce the answer is provided in JSON format. Fill in "xxx" with values ranging from 0.0 to 1.0, in increments of 0.1. The reasoning may include the starting point of thought, the process of elimination, or true statements, although these may not appear to be directly related to the answer at first glance. A value closer to 0.0 indicates a completely incorrect rationale, 0.5 indicates a neutral rationale such as the initial thought process or true statements that guide later guesses towards the answer, and a value closer to 1.0 denotes a correct or relevant rationale for the question. Please just fill the "xxx" parts and only return the JSON format. If a sentence is repetitive (appeared before), then give 0.0.\n\nQuestion: {question}\nOptions: [{choice}]\nHint: {hint}\nAnswer: {answer}\n\n{rationales_dict}"""
                    else:
                        source_file = os.path.join(ck_image_file, 'image.png')
                        destination_file = os.path.join(subfolder, 'image.png')
                        shutil.copy(source_file, destination_file)
                        input_text = f"""There exists a set comprising Image, Options, Hint, and Answer for a Question. The reasoning process used to deduce the answer is provided in JSON format. Fill in "xxx" with values ranging from 0.0 to 1.0, in increments of 0.1. The reasoning may include the starting point of thought, the process of elimination, or true statements, although these may not appear to be directly related to the answer at first glance. A value closer to 0.0 indicates a completely incorrect rationale, 0.5 indicates a neutral rationale such as the initial thought process or true statements that guide later guesses towards the answer, and a value closer to 1.0 denotes a correct or relevant rationale for the question. Please just fill the "xxx" parts and only return the JSON format. If a sentence is repetitive (appeared before), then give 0.0.\n\nQuestion: {question}\nOptions: [{choice}]\nHint: {hint}\nAnswer: {answer}\n\n{rationales_dict}"""
                    final_input_path = os.path.join(subfolder, 'input_for_llm.txt')
                    with open(final_input_path, 'w') as file:
                        file.write(input_text)
                # for finished signal
                final_input_action_path = os.path.join(i_action_folder_path,"preprocessing_finished_signal.txt")
                with open(final_input_action_path, 'w'):
                    pass
            
            print("\n\n All preprocessing is finished!! \n\n")
        else:
            time.sleep(30)

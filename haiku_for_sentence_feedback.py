import base64
import requests
from PIL import Image
import tiktoken
import os
import shutil
import openai
import time
import httpx
from anthropic import AnthropicVertex
import anthropic

directory = f"./RL_models/questions"
target_folder = "/p0"  # Change target_folder from p0 to p3
api_idx = 0  # Change api_idx from 0 to 3
api_key_list = [
    "YOUR_API_KEY_1",
    "YOUR_API_KEY_2",
    "YOUR_API_KEY_3",
    "YOUR_API_KEY_4"
    # Replace these with your own API keys
]

client = anthropic.Anthropic(
    api_key=api_key_list[api_idx],
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_number(dir_name):
    try:
        return int(dir_name)
    except ValueError:
        return float('inf')  # Sort non-numeric names to the end

def read_file_if_exists(directory, filename):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            return file.read()
    return None

def compare_files(base_dir, dirs, filename):
    base_file_content = read_file_if_exists(base_dir, filename)
    total_check = []
    for dir_path in dirs:
        compare_file_content = read_file_if_exists(dir_path, filename)
        if base_file_content != compare_file_content:
            total_check.append(False)
        else:
            total_check.append(True)
    return total_check

def process_message(max_token, user_prompt, image_path=None):
    if image_path and os.path.isfile(image_path):
        base64_image = encode_image(image_path)
        image_media_type = "image/png"
        message = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=max_token,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": image_media_type,
                                        "data": base64_image,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": user_prompt
                                }
                            ],
                        }
                    ],
                )
    else:
        message = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=max_token,
                    temperature=0.0,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
    
    return message

subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

sorted_subdirs = sorted(subdirs, key=extract_number)

encoding = tiktoken.encoding_for_model("gpt-4")

for topdir in sorted_subdirs:
    process_dir = os.path.join(directory, topdir + target_folder)
    process_subdirs = [d for d in os.listdir(process_dir) if os.path.isdir(os.path.join(process_dir, d))]
    sorted_process_subdirs = sorted(process_subdirs, key=extract_number)
    if topdir == "0":
        for subdir in sorted_process_subdirs:
            subdir_path = os.path.join(process_dir, subdir)
            if os.path.isdir(subdir_path):
                output_path = os.path.join(subdir_path, "output_from_llm.txt")
                if os.path.isfile(output_path):
                    continue
                else:
                    text_file_path = os.path.join(subdir_path, "input_for_llm.txt")
                    with open(text_file_path, 'r') as file:
                        user_prompt = file.read()
                    check_file = os.path.join(subdir_path, "rationale_from_mm_cot.txt")
                    with open(check_file, 'r') as file:
                        check_contents = file.read()
                    if check_contents != "{\n}":
                        max_token = len(encoding.encode(user_prompt)) * 2
                        image_path = os.path.join(subdir_path, 'image.png')
                        try:
                            message = process_message(max_token, user_prompt, image_path)
                        except Exception as e:
                            error_message = str(e)
                            if "overloaded_error" in error_message:
                                print("Server is overloaded. Trying again.")
                                message = process_message(max_token, user_prompt, image_path)
                            else:
                                print("Server error occurred. Trying again:", error_message)
                                message = process_message(max_token, user_prompt, image_path)
                        with open(output_path, 'w') as file:
                            file.write(message.content[0].text)
                        print(f"Processing folder {subdir_path}")
    else:
        for subdir in sorted_process_subdirs:
            subdir_path = os.path.join(process_dir, subdir)
            if os.path.isdir(subdir_path):
                output_path = os.path.join(subdir_path, "output_from_llm.txt")
                if os.path.isfile(output_path):
                    continue
                compare_list = []
                filename = 'rationale_from_mm_cot.txt'
                for i in range(sorted_subdirs.index(topdir)):
                    tmp_path = os.path.join(directory, sorted_subdirs[i] + target_folder + "/" + subdir)
                    compare_list.append(tmp_path)
                check_list = compare_files(subdir_path, compare_list, filename)
                if True not in check_list:  # if different
                    print(f"Processing folder {subdir_path}")
                    output_path = os.path.join(subdir_path, "output_from_llm.txt")
                    text_file_path = os.path.join(subdir_path, "input_for_llm.txt")
                    with open(text_file_path, 'r') as file:
                        user_prompt = file.read()
                    max_token = len(encoding.encode(user_prompt)) * 2
                    image_path = os.path.join(subdir_path, 'image.jpeg')
                    exception_path = os.path.join(subdir_path, "exception.txt")
                    try:
                        message = process_message(max_token, user_prompt, image_path)
                    except Exception as e:
                        error_message = str(e)
                        if "overloaded_error" in error_message:
                            print("Server is overloaded. Trying again.")
                            message = process_message(max_token, user_prompt, image_path)
                        else:
                            print("Server error occurred. Trying again:", error_message)
                            message = process_message(max_token, user_prompt, image_path)
                    with open(output_path, 'w') as file:
                        file.write(message.content[0].text)
                else:
                    print(f"Copied folder {subdir_path}")
                    original_path = os.path.join(compare_list[check_list.index(True)], "output_from_llm.txt")
                    shutil.copyfile(original_path, output_path)


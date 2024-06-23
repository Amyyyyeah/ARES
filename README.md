# ARES: Alternating Reinforcement Learning and Supervised Fine-Tuning for Enhanced Multi-Modal Chain-of-Thought Reasoning Through Diverse AI Feedback

<img src="https://github.com/Amyyyyeah/ARES/blob/main/pipeline.jpg" width="850px" height="300px" title="Overview of ARES" alt="server2"></img><br/>

## Requirements

Install all required python dependencies:

```
pip install -r requirements.txt
```

## Datasets

Download the dataset from the following repository:

```
https://github.com/lupantech/ScienceQA/tree/main/data
```
The vision features (detr, resnet, clip, vit) are available at https://huggingface.co/cooelf/vision_features/tree/main

Alternatively, you may download the extracted vision features (detr, resnet, clip) from [vision_features](https://drive.google.com/file/d/13B0hc_F_45-UlqPLKSgRz-ALtFQ8kIJr/view?usp=share_link) and unzip the files under `vision_features`

## Extract Features (optional)

The processed vision features for ScienceQA are available at https://huggingface.co/cooelf/vision_features/tree/main. 

The following instructions show how we obtain those features.

Download the image files from [Google Drive](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev?usp=sharing) and unzip all the images (train, dev, test) in the same folder (). The structure should be:

```
images
├── 1
│   └── image.png
├── 2
│   └── image.png
├── 3
│   └── image.png
├── 5
│   └── image.png
├── 7
│   └── image.png
```

Run ```extract_features.py --data_root images --output_dir vision_features --img_type vit```

If you hope to use your own images, please structure those images in the way above, or modify the script ```extract_features.py```.

## Extract Captions (optional)

The processed captions for ScienceQA are available at ```data/instruct_captions.json```. 

The following instructions show how we obtain those features.

Intall lavis and prepare Vicuna weights to use InstructBLIP for caption extraction.

https://github.com/salesforce/LAVIS/tree/f982acc73288408bceda2d35471a8fcf55aa04ca/projects/instructblip

Assume that the images are stored in the ```images``` folder. 

```
python extract_caption.py
```

## Instructions 
**Our ARES Training consists of the following three steps: RL, SFT, and LoRA.**

Our trained models are available at https://huggingface.co/JCAC/ARES/~. To use our trained models for testing, please place them under the models folder.
(If using the A-OKVQA dataset, change the following paths to the A-OKVQA dataset path in the code and bash arguments.)

✔️ Before following the steps, you need to obtain the Claude 3 Haiku API keys.

### # Our ARES Training Steps:
### [Step 1] Reinforcement Learning (RL)
* We use 4 NVIDIA A100 GPUs with 80GB memory for RL training.
```
# Base - RL training
accelerate launch run_mm_cot_rl.py \
    --data_root scienceqa_data/ScienceQA/data \
    --caption_file scienceqa_data/instruct_captions.json \
    --user_msg rationale --img_type vit \
    --use_caption --use_generate --prompt_format QCM-E \
    --seed 42 \
    --model "declare-lab/flan-alpaca-base" \
    --model_type base \
    --base_model_dir ./models/mm-cot-base-rationale \
    --ref_model ./models/mm-cot-base-rationale \
    --k_actions 4 --train_split train \
    --continue_train False \
    --do_sample True \
    --bs 64 --output_len 512 \
    --rl_batch_size 8 \
    --init_kl_coef 0.0001 --top_k 50 \
    --rl_epochs 10 --lr 2e-5 --clip_range 0.2 --epochs 1 --ga_step 8 --gamma 1.0 --adv_normalization True
```

* If there is a message 'Rationale Finished. Waiting for the feedback', Sentence-level nuanced feedback is needed.
    - First, copy ```./RL_models/{current_model_path}/questions/*``` to the ```./preprocessing_after_RL``` path.
    - So, run ```./preprocessing_after_RL/processing_sentence_level_feedback.sh``` for preprocessing to get the feedback, and get the sentence-level nuanced feedback by running ```./haiku.py```.
    - After finishing getting feedback, copy the questions folder back to ```./RL_models/{current_model_path}```.
    - Then, create a file named ```llm_done.txt``` in the path ```./RL_models/{current_model_path}/questions/0/, RL_models/{current_model_path}/questions/1/, RL_models/{current_model_path}/questions/2/, and RL_models/{current_model_path}/questions/3/```.

      (use the command ```touch ./RL_models/{current_model_path}/questions/{0,1,2,3}/llm_done.txt```).
      
```
# Base - Generate predictions_ans_*.json (Use 1 NVIDIA A100 GPU)
CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root scienceqa_data/ScienceQA/data \
    --model "declare-lab/flan-alpaca-base" \
    --caption_file scienceqa_data/instruct_captions.json \
    --user_msg rationale --img_type vit \
    --bs 16 --eval_bs 8 --epoch 20 --lr 8e-5 --output_len 512 \
    --use_caption --use_generate --final_eval --prompt_format QCM-E \
    --output_dir experiments \
    --seed 42 \
    --evaluate_dir ./RL_models/base_neutral0.5_k4_rlb8_cl0.2_rle10_lr2e-05_vlr1.0_g1.0_l0.95_fGPT4V_seed42_kl0.0001_ga8_dosampleTrue_advTrue_tk50_ref/0
```
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
# Large - RL training
accelerate launch run_mm_cot_rl.py \
    --data_root scienceqa_data/ScienceQA/data \
    --caption_file scienceqa_data/instruct_captions.json \
    --user_msg rationale --img_type vit \
    --use_caption --use_generate --prompt_format QCM-E \
    --seed 42 \
    --model "declare-lab/flan-alpaca-large" \
    --model_type large \
    --base_model_dir ./models/mm-cot-large-rationale \
    --ref_model ./models/mm-cot-large-rationale \
    --k_actions 4 --train_split train \
    --continue_train False \
    --do_sample True \
    --bs 32 --output_len 512 \
    --rl_batch_size 2 \
    --init_kl_coef 0.0001 --top_k 50 \
    --rl_epochs 5 --lr 2e-5 --clip_range 0.2 --epochs 1 --ga_step 16 --gamma 1.0 --adv_normalization False

# Large - Generate predictions_ans_*.json (Use 4 NVIDIA A100 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --data_root scienceqa_data/ScienceQA/data \
    --model "declare-lab/flan-alpaca-large" \
    --caption_file scienceqa_data/instruct_captions.json \
    --user_msg rationale --img_type vit \
    --bs 2 --eval_bs 4 --epoch 50 --lr 5e-5 --output_len 512 \
    --use_caption --use_generate --prompt_format QCM-E \
    --output_dir experiments \
    --seed 42 \
    --evaluate_dir ./RL_models/large_neutral0.5_k4_rlb2_cl0.2_rle5_lr2e-05_vlr1.0_g1.0_l0.95_fGPT4V_seed42_kl0.0001_ga16_dosampleTrue_advFalse_tk50_ref/0
```

### [Step 2] Supervised Fine-Tuning (SFT)
* We request correction feedback from advanced AI (Teacher) for sentences containing errors after the RL process. To get correction feedback from Haiku of Claude 3, you need to follow the three steps below first and then train using SFT with the correction file.

**# Getting Correction Feedback**

[1] Run the following command using Python:
```
python ./preprocessing_after_RL/remove_sentence.py --file_path ./RL_models/{current_model}/{action}/prediction_ans_train.json --tokenizer ./RL_models/{current_model}/{action}
```

[2] Run ```./preprocessing_for_correction_feedback.py``` for the preprocessing.

[3] Run ```./haiku_for_correction_feedback.py``` to get the correction feedback and save the correction file.

    


**# After finishing getting feedback, enter the correction file path in the --correction_file.**
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root data/ScienceQA/data \
    --correction True --correction_file scienceqa/correction.json \
    --caption_file data/instruct_captions.json \
    --model ./RL_models/base_neutral0.5_k4_rlb8_cl0.2_rle10_lr2e-05_vlr1.0_g1.0_l0.95_fGPT4V_seed42_kl0.0001_ga8_dosampleTrue_advTrue_tk50_ref/0 \
    --user_msg rationale --img_type vit \
    --bs 8 --eval_bs 8 --epoch 20 --lr 8e-5 --output_len 512 \
    --use_caption --use_generate --final_eval --prompt_format QCM-E \
    --output_dir experiments
```

### [Step 3] LoRA adapter
```
CUDA_VISIBLE_DEVICES=0 python run_mm_cot_lora.py \
    --correction False \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model {correction_trained_model_path under experiments} \
    --user_msg answer --img_type vit \
    --bs 16 --eval_bs 8 --epoch 20 --lr 8e-5 --output_len 64 \
    --use_caption --use_generate --final_eval --prompt_format QCM-A \
    --seed 42 \
    --eval_le {correction_trained_model_path under experiments}/predictions_ans_eval.json \
    --test_le {correction_trained_model_path under experiments}/predictions_ans_test.json \
    --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
```
* See the results in {correction_trained_model_path under experiments}/{lora_trained_path}/prediction_ans_test.json.
  
## Citing ARES

```

```

## License

This project is licensed under the MIT License.

## Acknowledgement

Some parts of our code are adapted from [ScienceQA](https://github.com/lupantech/ScienceQA), [Transformers](https://github.com/huggingface/transformers), [pytorch-image-models](https://github.com/huggingface/pytorch-image-models).

Additionally, we have referenced the code from [MM-CoT](https://github.com/amazon-science/mm-cot/tree/main).

# ARES: Alternating Reinforcement Learning and Supervised Fine-Tuning for Enhanced Multi-Modal Chain-of-Thought Reasoning Through Diverse AI Feedback

<img src="https://github.com/Amyyyyeah/EMNLP24_ARES/blob/main/ares.jpg" width="850px" height="500px" title="Overview of ARES" alt="server2"></img><br/>

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

### [Step 1] RL training (4 NVIDIA A100 GPUs with 80GB memory) - RL.sh

```
# Base - RL training
accelerate launch run_mm_cot_rl.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
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

# Base - Generate predictions_ans_*.json (Use 1 NVIDIA A100 GPU)
CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root data/ScienceQA/data \
    --model "declare-lab/flan-alpaca-base" \
    --caption_file data/instruct_captions.json \
    --user_msg rationale --img_type vit \
    --bs 16 --eval_bs 8 --epoch 20 --lr 8e-5 --output_len 512 \
    --use_caption --use_generate --final_eval --prompt_format QCM-E \
    --output_dir experiments \
    --seed 42 \
    --evaluate_dir /fs/scratch/PAS2138/mm_cot/base_neutral0.5_k4_rlb8_cl0.2_rle10_lr2e-05_vlr1.0_g1.0_l0.95_fGPT4V_seed42_kl0.0001_ga8_dosampleTrue_advTrue_tk50_ref/1

# Large - RL training
accelerate launch run_mm_cot_rl.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --user_msg rationale --img_type vit \
    --use_caption --use_generate --prompt_format QCM-E \
    --seed 42 \
    --model "declare-lab/flan-alpaca-large" \
    --model_type base \
    --base_model_dir ./models/mm-cot-large-rationale \
    --ref_model ./models/mm-cot-large-rationale \
    --k_actions 4 --train_split train \
    --continue_train False \
    --do_sample True \
    --bs 32 --output_len 512 \
    --rl_batch_size 2 \
    --init_kl_coef 0.0001 --top_k 50 \
    --rl_epochs 5 --lr 2e-5 --clip_range 0.2 --epochs 1 --ga_step 16 --gamma 1.0 --adv_normalization False

# Large - Generate predictions_ans_*.json (Use 4 NVIDIA A100 GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --data_root data/ScienceQA/data \
    --model "declare-lab/flan-alpaca-large" \
    --caption_file data/instruct_captions.json \
    --user_msg rationale --img_type vit \
    --bs 2 --eval_bs 4 --epoch 50 --lr 5e-5 --output_len 512 \
    --use_caption --use_generate --prompt_format QCM-E \
    --output_dir experiments \
    --seed 42 \
    --evaluate_dir /fs/scratch/PAS2138/mm_cot/haiku_feedback/large_neutral0.5_k4_rlb2_cl0.2_rle5_lr2e-05_vlr1.0_g1.0_l0.95_fGPT4V_seed42_kl0.0001_ga16_dosampleTrue_advFalse_tk50_ref/1
    
```

### [Step 2] SFT

Our trained models are available at https://huggingface.co/JCAC/ARES/~. To use our trained models, please put the them under the ```models``` folder.

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root data/ScienceQA/data \
    --correction True --correction_file scienceqa/aokvqa_2e5_e10_t_3.json \
    --caption_file data/instruct_captions.json \
    --model /fs/scratch/PAS2138/mm_cot/base_neutral0.5_k4_rlb8_cl0.2_rle10_lr2e-05_vlr1.0_g1.0_l0.95_fGPT4V_seed42_kl0.0001_ga8_dosampleTrue_advTrue_tk50_ref/2 \
    --user_msg rationale --img_type vit \
    --bs 8 --eval_bs 8 --epoch 20 --lr 8e-5 --output_len 512 \
    --use_caption --use_generate --final_eval --prompt_format QCM-E \
    --output_dir experiments

# answer inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_central.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-large \
    --user_msg answer --img_type vit \
    --bs 4 --eval_bs 8 --epoch 50 --lr 5e-5 --output_len 64  \
    --use_caption --use_generate --prompt_format QCMG-A \
    --output_dir experiments \
    --eval_le experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs8_op512_ep50/predictions_ans_eval.json \
    --test_le experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs8_op512_ep50/predictions_ans_test.json \
    --evaluate_dir models/mm-cot-large-answer
```

### [Step 3] LoRA

## Citing ARES

```

```

## License

This project is licensed under the MIT License.

## Acknowledgement

Part of our codes are adapted from [ScienceQA](https://github.com/lupantech/ScienceQA), [Transformers](https://github.com/huggingface/transformers), [pytorch-image-models](https://github.com/huggingface/pytorch-image-models) [MM-CoT](https://github.com/amazon-science/mm-cot/tree/main).

We thank [Pan Lu](https://lupantech.github.io/) for providing parameter size for ScienceQA baselines.


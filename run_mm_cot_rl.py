import os
from os.path import join as osp
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import re
import json
import argparse
import random
from copy import deepcopy
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from transformers import AdamW
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import RemoveColumnsCollator
from model import T5ForMultimodalGeneration, T5ForMultimodalGenerationValueFunction, FixedKLController
from utils_data import img_shape, load_data_std, load_data_img, ScienceQADatasetImgWithIndex # ScienceQADatasetImg
from utils_prompt import *
from utils_evaluate import get_scores
from torch.utils.data import DataLoader
from functools import partial
from torch.utils.data.distributed import DistributedSampler
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import nltk
import evaluate
import time
from buffer import PPORLElement, PPORLVisionElement, PPORLBatchSampler, PPORLVisionDataset, ppo_rl_collate_fn
from utils import check_eos, logprobs_of_labels
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from accelerate import Accelerator
from torchtyping import TensorType
import copy
import shutil


def log_message(message, is_main_process):
    if is_main_process:
        console.log(message)  


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_global_statistics(xs: torch.Tensor, group=None) -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    """
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    dist.all_reduce(sum_and_count, dist.ReduceOp.SUM, group=group)
    global_sum, count = sum_and_count
    global_mean = global_sum / count
    sum_var = torch.sum((xs - global_mean) ** 2)
    dist.all_reduce(sum_var, dist.ReduceOp.SUM, group=group)
    global_var = sum_var / count
    return global_mean, global_var, count


def whiten(xs: torch.Tensor, shift_mean=True, distributed=True, group=None) -> torch.Tensor:
    """Whitens values"""
    if distributed and dist.is_initialized():
        mean, var, _ = get_global_statistics(xs, group=group)
    else:
        var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='declare-lab/flan-alpaca-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=512)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain', 'miniminitrain', 'hardtrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    
    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet','vit'], help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--base_model_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-E', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=3, help='the number of receiving samples') 
    parser.add_argument('--vf_coef', type=float, default=1.0)
    parser.add_argument('--rl_batch_size', type=int, default=6, help='batch size of ppo')
    parser.add_argument('--rl_epochs', type=int, default=3, help='epochs of ppo for a dataset')
    parser.add_argument('--clip_range', type=float, default=0.2, help='clip range of ppo')
    parser.add_argument('--k_actions', type=int, default=3, help='k different generation for a sample')
    parser.add_argument('--neutral_score', type=float, default=0.5, help='the neutral score for LLM feedback')
    parser.add_argument('--gamma', type=float, default=0.99, help='GAE gamma')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--algo', type=str, default='ppo', help='algorithm type')
    parser.add_argument('--log_term', type=int, default=200)
    parser.add_argument('--ref_model', type=str, default=None)
    parser.add_argument('--feedback', type=str, default='GPT4V')
    parser.add_argument('--model_type', type=str, default='base')
    parser.add_argument('--continue_train', type=str2bool, default=False)
    parser.add_argument('--init_kl_coef', type=float, default=0.05)
    parser.add_argument('--ga_step', type=int, default=4)
    parser.add_argument('--do_sample', type=str2bool, default=False)
    parser.add_argument('--adv_normalization', type=str2bool, default=True)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=20)
    args = parser.parse_args()
    return args


        
class RL_T5Trainer:
    def __init__(self, dataframe, args) -> None:
        self.dataframe = dataframe
        self.args = args
        self.num_gpus = torch.cuda.device_count()
        self.start_outer_epoch = 0
        if self.args.base_model_dir is not None and args.continue_train == True:
            last_part = self.args.base_model_dir.split('/')[-1]
            # Convert it to an integer
            try:
                self.start_outer_epoch = int(last_part) + 1
            except ValueError:
                # Handle the case where the last part is not a number
                raise(f"Warning: The last part of the path '{last_part}' is not an integer.") 

        self.train_data_len = len(dataframe['qids']['train']) 
        self.image_feature_shape = dataframe['image_features'].shape[1:]
        # oi = original question index, ci = changed question index  
        self.train_oi_to_ci = {}
        self.train_ci_to_oi = {}


        qids = dataframe['qids']
        train_qids = qids['train']

        temp_data = [qid for qid in train_qids]
        idx = 0
        for qid in temp_data:
            self.train_oi_to_ci[int(qid)] = idx
            self.train_ci_to_oi[idx] = int(qid)
            idx += 1

        self.rl_batch_size = args.rl_batch_size
        self.rl_epochs = args.rl_epochs
        self.k_actions = args.k_actions
        self.neutral_score = args.neutral_score
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.clip_range = args.clip_range
        self.gradient_accumulation_steps = args.ga_step

        self._setup()
    
    def _setup(self):
        
        #self.base_dir = f"{os.getenv('HOME')}/mm_cot_rl_rle{self.rl_epochs}/"
        #self.base_dir = f"{os.getenv('HOME')}/mm_cot_rl/"
        

        if self.args.ref_model is not None:
            self.base_dir = f'./RL_models/{self.args.model_type}_neutral{self.args.neutral_score}_k{self.args.k_actions}_rlb{self.args.rl_batch_size}_cl{self.args.clip_range}_rle{self.args.rl_epochs}_lr{self.args.lr}_vlr{self.args.vf_coef}_g{self.args.gamma}_l{self.args.gae_lambda}_f{self.args.feedback}_seed{self.args.seed}_kl{self.args.init_kl_coef}_ga{self.args.ga_step}_dosample{self.args.do_sample}_adv{self.args.adv_normalization}_tk{self.args.top_k}_ref/'
        else:
            self.base_dir = f'./RL_models/{self.args.model_type}_neutral{self.args.neutral_score}_k{self.args.k_actions}_rlb{self.args.rl_batch_size}_cl{self.args.clip_range}_rle{self.args.rl_epochs}_lr{self.args.lr}_vlr{self.args.vf_coef}_g{self.args.gamma}_l{self.args.gae_lambda}_f{self.args.feedback}_seed{self.args.seed}_kl{self.args.init_kl_coef}_ga{self.args.ga_step}_dosample{self.args.do_sample}__adv{self.args.adv_normalization}_tk{self.args.top_k}_no_ref/'

        from accelerate import DistributedDataParallelKwargs

        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.accelerator = Accelerator()

        console.log(f"""[Base_dir]: {self.base_dir}\n""")
        self.question_dir = osp(self.base_dir, 'questions')

        self.exception_num = 0

        if not os.path.exists(self.question_dir):
            os.makedirs(self.question_dir, exist_ok=True)

        torch.manual_seed(self.args.seed)  # pytorch random seed
        np.random.seed(self.args.seed)  # numpy random seed
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(self.args.seed)       # for multi-GPU.
        
        if self.args.base_model_dir == None:
            raise ValueError("base_moel_dir should be not none")
        
        self.args.model = self.args.base_model_dir
        tokenizer = AutoTokenizer.from_pretrained(self.args.base_model_dir)

        # tokenizer = T5Tokenizer.from_pretrained('./ppo_value/neutral0.5_k1_rlb4_rlbv64_cl0.2_rle2_lr2e-05_vlr5e-05_g0.99_l0.95_tk40_t2.0')
        log_message(f"""[Model]: Loading {self.args.model}...\n""", self.accelerator.is_main_process)
        log_message(f"[Data]: Reading data...\n", self.accelerator.is_main_process)

        problems = dataframe['problems']
        qids = dataframe['qids']
        train_qids = qids['train']
        test_qids = qids['test']
        val_qids = qids['val']

        if self.args.img_type is not None:
            patch_size = img_shape[self.args.img_type]
            model = T5ForMultimodalGeneration.from_pretrained(self.args.model, patch_size=patch_size) 
            if self.args.ref_model is not None:
                ref_model = T5ForMultimodalGeneration.from_pretrained(self.args.ref_model, patch_size=patch_size) 
            value_function = T5ForMultimodalGenerationValueFunction.from_pretrained(self.args.model, patch_size=patch_size) 

            name_maps = dataframe['name_maps'] 
            image_features = dataframe['image_features']
            train_set = ScienceQADatasetImgWithIndex(
                problems,
                train_qids,
                name_maps,
                tokenizer,
                self.args.input_len,
                self.args.output_len,
                self.args,
                image_features,
            )
            eval_set = ScienceQADatasetImgWithIndex(
                problems,
                val_qids,
                name_maps,
                tokenizer,
                self.args.input_len,
                self.args.output_len,
                self.args,
                image_features,
                self.args.eval_le,
            )
            test_set = ScienceQADatasetImgWithIndex(
                problems,
                test_qids,
                name_maps,
                tokenizer,
                self.args.input_len,
                self.args.output_len,
                self.args,
                image_features,
                self.args.test_le,
            )
        # TODO
        # else:
        #     model = T5ForConditionalGeneration.from_pretrained(self.args.model) 
        #     encoder_for_value_function = T5ForConditionalGeneration.from_pretrained(self.args.model) 

        #     train_set = ScienceQADatasetStd(
        #         problems,
        #         train_qids,
        #         tokenizer,
        #         self.args.input_len,
        #         self.args.output_len,
        #         self.args,
        #     )
        #     eval_set = ScienceQADatasetStd(
        #         problems,
        #         val_qids,
        #         tokenizer,
        #         self.args.input_len,
        #         self.args.output_len,
        #         self.args,
        #         self.args.eval_le,
        #     )
            
        #     test_set = ScienceQADatasetStd(
        #         problems,
        #         test_qids,
        #         tokenizer,
        #         self.args.input_len,
        #         self.args.output_len,
        #         self.args,
        #         self.args.test_le,
        #     )

        datacollator = DataCollatorForSeq2Seq(tokenizer)
        log_message(f"""[Model_Parameters]: {model.num_parameters()}\n""", self.accelerator.is_main_process)

        self._set_signature_columns_if_needed(model)
        removed_data_collator = RemoveColumnsCollator(
                    data_collator=datacollator,
                    signature_columns=self.signature_columns,
                    logger=None,
                    description='traning',
                    model_name=model,
                )


        self.dataloader = DataLoader(train_set, batch_size=self.args.bs, collate_fn=removed_data_collator, pin_memory=True)
        self.device = self.accelerator.device
        self.process_index = self.accelerator.state.process_index
        self.model = model
        self.ref_model = ref_model if self.args.ref_model is not None else deepcopy(self.model)
        self.ref_model.eval()  
        self.value_function = value_function
        self.kl_ctl = FixedKLController(self.args.init_kl_coef)
        
        self.tokenizer = tokenizer
        self.optimizer = self.create_optimizer(self.model, self.value_function)

        if self.start_outer_epoch != 0:
            #self.optimizer.load_state_dict(torch.load(osp(self.args.base_model_dir, "optimizer_state.pth")))
            self.value_function.load_state_dict(torch.load(osp(self.args.base_model_dir, "value_function.pth")))
            log_message(f"""Loading the value function {osp(self.args.base_model_dir, "value_function.pth")}\n""", self.accelerator.is_main_process)

        self.model, self.ref_model, self.value_function, self.dataloader, self.optimizer = \
            self.accelerator.prepare(self.model, self.ref_model, self.value_function, self.dataloader, self.optimizer)

        self.generate_kwargs = dict(
            do_sample=self.args.do_sample,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.args.output_len,
            top_k=self.args.top_k
            # top_p=self.args.top_p
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        log_message(f"""Device: {self.accelerator.device}""", self.accelerator.is_main_process)
        log_message(f"""State: {self.accelerator.state}""", self.accelerator.is_main_process)

    def _set_signature_columns_if_needed(self, model):
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(model.forward)
        self.signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        self.signature_columns += list(set(["label", "label_ids"] + ['labels'] + ['problem_indices']))
        

    def create_optimizer(self, model, value_function):
        all_named_parameters = list(model.named_parameters()) + list(value_function.named_parameters())
    
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS) + get_parameter_names(value_function, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
    

        param_groups = [
        {
            "params": [p for n, p in all_named_parameters if n in decay_parameters and p.requires_grad],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in all_named_parameters if n not in decay_parameters and p.requires_grad],
            "weight_decay": 0.0,
        },
        ]

        return AdamW(param_groups, lr=self.args.lr, betas=(0.9, 0.999), eps=1e-8, correct_bias=True)


    def collect_sample(self):
        self.model.eval()
        self.value_function.eval()
        self.ppo_rl_elements = {}

        for step, inputs in enumerate(self.dataloader):
            log_message(f"""Collect Sample Step: {step + 1} / {self.train_data_len // (self.args.bs * self.accelerator.state.num_processes) + 1}\n""", self.accelerator.is_main_process)
            #self.accelerator.free_memory()
            self.accelerator.wait_for_everyone()
            for k in range(self.k_actions):
                with torch.no_grad():
                    input_ids = inputs["input_ids"].to(self.device)
                    attention_mask = inputs["attention_mask"].to(self.device)
                    image_ids = inputs["image_ids"].to(self.device)
                    indices = inputs["problem_indices"].to(self.device)
                    indices = indices + k * self.train_data_len
                    actions = self.accelerator.unwrap_model(self.model).generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            image_ids=image_ids,
                            **self.generate_kwargs
                        )

                    device = actions.device

                    maxsize = max(map(len, actions))
                    padded_actions = [F.pad(action, (0, maxsize - len(action)), value=self.tokenizer.pad_token_id,) for action in actions]

                    padded_actions = torch.vstack(padded_actions).to(device)
                    decoder_attention_mask = padded_actions.not_equal(self.tokenizer.pad_token_id)
                    decoder_attention_mask[:, 0] = 1
                    
                    cur_outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_ids=image_ids,
                        decoder_input_ids=padded_actions,
                        decoder_attention_mask=decoder_attention_mask,
                    )

                    ref_outputs = self.ref_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_ids=image_ids,
                        decoder_input_ids=padded_actions,
                        decoder_attention_mask=decoder_attention_mask,
                    )

                    values = self.value_function(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_ids=image_ids,
                        decoder_input_ids=padded_actions,
                        decoder_attention_mask=decoder_attention_mask,
                    )
                    
                    logits = cur_outputs.logits
                    logprobs = logprobs_of_labels(logits[:, :-1, :], padded_actions[:, 1:])

                    ref_logits = ref_outputs.logits
                    ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], padded_actions[:, 1:])

                    attention_mask = padded_actions != self.tokenizer.pad_token_id
                    log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
                    kl = log_ratio.exp() - 1 - log_ratio
                    mean_kl_per_token = kl.mean()
                    mean_kl = kl.sum(1).mean()

                    logprobs = logprobs.cpu()
                    ref_logprobs = ref_logprobs.cpu()
                    input_ids = input_ids.cpu()
                    image_ids = image_ids.cpu()
                    actions = padded_actions.cpu()
                    values = values.cpu()[:, :-1]
                    indices = indices.cpu().tolist()

                    n_samples: int = actions.shape[0]

                    ends = attention_mask.sum(1) + 1
                
                    all_values = [values[ix, : ends[ix]] for ix in range(n_samples)]
                    all_logprobs = [logprobs[ix, : ends[ix]] for ix in range(n_samples)]
                    
                    kl_penalty = self.kl_ctl.value * -log_ratio.cpu()
                    kl_penalty = [xs[: ends[ix]] for ix, xs in enumerate(kl_penalty)]    
                    for sample_idx in range(n_samples):
                        rewards = kl_penalty[sample_idx]
                        self.ppo_rl_elements[indices[sample_idx]] =\
                            PPORLVisionElement(
                                input_ids=input_ids[sample_idx],
                                image_ids=image_ids[sample_idx],
                                actions=actions[sample_idx],
                                logprobs=all_logprobs[sample_idx],
                                values=all_values[sample_idx],
                                rewards=rewards # sentence-based reward feedback will be added
                            )

    def export_file(self):
        log_message(f"""Start exporting rationale\n""", self.accelerator.is_main_process)
        
        # make sure each problem (index) is processed only once and check the number of sentences.
        self.sentence_indices = {} 

        for index, e in self.ppo_rl_elements.items():
            self.sentence_indices[index] = []
            decoded_sentences_list = ['{']
            encoded_sentences = e.actions.unsqueeze(0)
            start = 0
            for col_idx in range(encoded_sentences.shape[1]):
                mask = check_eos(encoded_sentences, col_idx)
                
                if mask[0] == True:
                    self.sentence_indices[index].append(col_idx - 1)
                    
                    decoded_sentence = self.tokenizer.decode(encoded_sentences[0][start:col_idx+1], skip_special_tokens=True)
                    if len(decoded_sentence) > 0 and decoded_sentence[0] == 'n':
                        decoded_sentence = decoded_sentence[1:]
                    decoded_sentences_list.append(f'"{decoded_sentence}": xxx,')
                    start = col_idx + 1
            
            if len(decoded_sentences_list) > 1:
                decoded_sentences_list[1] = decoded_sentences_list[1].replace("Solution: ", "").strip()
                decoded_sentences_list[1] = decoded_sentences_list[1].replace("Solution:", "").strip()
            decoded_sentences_list[-1] = decoded_sentences_list[-1].rstrip(",")
            decoded_sentences_list.append('}')

            parent_dir = index // self.train_data_len
            original_index = self.train_ci_to_oi[index%self.train_data_len]
            folder_path = osp(osp(osp(self.question_dir, str(parent_dir)), f'p{self.process_index}'), str(original_index))
            
            os.makedirs(folder_path, exist_ok=True)
            
            with open(osp(folder_path, 'rationale_from_mm_cot.txt'), 'w', encoding='utf-8') as f:
                for sentence in decoded_sentences_list:
                    f.write(sentence + '\n')


    def wait_feedback(self):
        num_exception = 0
        total = 0
        while True:
            flag = True
            for llm_process_check in self.llm_process_check_list:
                flag = flag and os.path.exists(llm_process_check)

            if flag:
                check_path = f'p{self.process_index}'
                for sub_dir_name in os.listdir(self.question_dir):
                    sub_dir_path = osp(self.question_dir, sub_dir_name)
                    if os.path.isdir(sub_dir_path):
                        for root, _, _ in os.walk(sub_dir_path):
                            if check_path not in root: 
                                continue

                            has_exception = False
                            scores = []
                            rationale_num_sentences = 0 
                            output_scores = []
                            exeption_str = 'None'
                            txt_content_output = 'None'
                            if os.path.exists(osp(root, "rationale_from_mm_cot.txt")):
                                with open(osp(root, "rationale_from_mm_cot.txt"), 'r') as f:
                                    txt_content = f.read()
                                    lines = txt_content.strip().split('\n')
                                    for line in lines:
                                        for colon_idx in reversed(range(len(line))):
                                            if line[colon_idx] == ':':
                                                rationale_num_sentences += 1
                                                break
                            else:
                                continue

                                            
                            if os.path.exists(osp(root, 'output_from_llm.txt')):
                                with open(osp(root, 'output_from_llm.txt'), 'r') as ofile:
                                    txt_content_output = ofile.read()
                                    matches = re.findall(r'{.*?}', txt_content_output, re.DOTALL)
                                    if not matches:
                                        has_exception = True
                                        exeption_str = f"Exception in Folder: {root} - Format is not matched.\n"
                                    else:
                                        for json_str in matches:
                                            lines_output = json_str.strip().split('\n')
                                            for line in lines_output:
                                                for colon_idx in reversed(range(len(line))):
                                                    if line[colon_idx] == ':':
                                                        value = line[colon_idx+1:].strip(',').strip(' ').strip('}')
                                                        output_scores.append(value)
                                                        break
                                
                            if not has_exception:
                                if len(output_scores) != rationale_num_sentences:
                                    has_exception = True
                                    exeption_str = f"Exception in Folder: {root} - Mismatch in number of sentences.\n"
                                
                                for score in output_scores:
                                    if has_exception: break
                                    if value == 'xxx':
                                        has_exception = True
                                        exeption_str = f"Exception in Folder: {root} - Value is not 'xxx' for sentence: \n{txt_content_output}\n"
                                        continue

                                    try:
                                        float_score = float(score)
                                        if float_score is None or float_score < 0 or float_score > 1:
                                            raise ValueError
                                    except ValueError:
                                        has_exception = True
                                        exeption_str = f"Exception in Folder: {root} - Invalid score for sentence: \n{txt_content_output}\n"

                                    if not has_exception:
                                        scores.append(score)

                            if not has_exception and len(scores) == 0:
                                has_exception = True
                                exeption_str = f"Exception in Folder: {root} - The number of scores is 0\n"


                            if os.path.exists(osp(root, 'score.txt')):
                                os.remove(osp(root, 'score.txt'))
                            
                            if has_exception:
                                path_for_whole_exceptions = osp(os.path.dirname(self.base_dir), 'exceptions', str(self.exception_num))
                                if not os.path.exists(path_for_whole_exceptions):
                                    os.makedirs(path_for_whole_exceptions)

                                with open(osp(path_for_whole_exceptions,'exception.txt'), 'w') as exfile:
                                    exfile.write(f"{exeption_str}")

                                shutil.copy(osp(root, 'output_from_llm.txt'), path_for_whole_exceptions)
                                shutil.copy(osp(root, "rationale_from_mm_cot.txt"), path_for_whole_exceptions)
                                self.exception_num += 1
                                num_exception += 1

                            if not has_exception and scores:
                                with open(osp(root, 'score.txt'), 'w') as scorefile:
                                    for score in scores:
                                        scorefile.write(f"{score}\n")
                            total += 1
                break
            else:
                time.sleep(5) 


        console.log(f"""The number of exception.txt files: {num_exception} / {total}\n""")
        return

    def extract_feedback(self):      
        additional_num_exception = 0
        check_path = f'p{self.process_index}'
        for sub_dir_name in os.listdir(self.question_dir):
            sub_dir_path = osp(self.question_dir, sub_dir_name)
            if os.path.isdir(sub_dir_path):
                for root, _, _ in os.walk(sub_dir_path):
                    if check_path not in root: 
                        continue
                    file_path_score = osp(root, 'score.txt')
                    file_path_rationale = osp(root, 'rationale_from_mm_cot.txt')
                    if os.path.exists(file_path_rationale):
                        # Split the path to get all directories and the filename
                        path_parts = file_path_score.split(os.sep)

                        # Assuming the format is always some_directory/number1/px/number2/scores.txt
                        k_action_dir = int(path_parts[-4])
                        original_question_id = int(path_parts[-2])
                        changed_id = self.train_oi_to_ci[original_question_id]
                        changed_id = changed_id +  k_action_dir * self.train_data_len

                        if changed_id not in self.sentence_indices:
                            continue

                        if not os.path.exists(file_path_score):
                            additional_num_exception += 1 
                            del self.sentence_indices[changed_id]
                            del self.ppo_rl_elements[changed_id]
                            continue
                        
                        with open(file_path_score, 'r') as f:
                            scores = []
                            for line in f:
                                score = float(line.strip()) - self.neutral_score
                                scores.append(score)
                            
                            if len(scores) == 0 or len(scores) != len(self.sentence_indices[changed_id]):
                                additional_num_exception += 1 
                                del self.sentence_indices[changed_id]
                                del self.ppo_rl_elements[changed_id]
                                continue

                            for i in range(len(scores)):
                                score = scores[i]
                                pos = self.sentence_indices[changed_id][i]
                                self.ppo_rl_elements[changed_id].rewards[pos] += score # add a sentence reward
                    
        console.log(f"""additional_num_exception: {additional_num_exception} in {self.accelerator.state.process_index} process""")

    def loss(self,
        logprobs: TensorType["batch_size", "response_size"],
        values: TensorType["batch_size", "response_size"],
        old_logprobs: TensorType["batch_size", "response_size"],
        advantages: TensorType["batch_size", "response_size"],
        returns: TensorType["batch_size", "response_size"],
        mask: TensorType["batch_size", "response_size"],
    ):
        n = mask.sum()
        vf_loss = torch.sum((values - returns) ** 2 * mask) / n
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.clip_range,
            1.0 + self.clip_range,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

        loss = pg_loss + self.args.vf_coef * vf_loss

        stats = dict( 
            total_loss=loss.item(),
            policy_loss=pg_loss.item(),
            value_loss=vf_loss.item(),   
            clipfrac=pg_clipfrac.item()
        )

        return loss, stats

    def get_advantages_and_returns(
        self,
        values: TensorType["batch_size", "response_size"],
        rewards: TensorType["batch_size", "response_size"],
        response_length: int,
        use_whitening: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Args:
            values: Tensor of shape (batch_size, response_size)
            rewards: Tensor of shape (batch_size, response_size)
            response_length: Length of the response sequence
            use_whitening: Whether to use whitening (ie. normalize advantages) or not
        """
        lastgaelam = 0
        advantages_reversed = []

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.gae_lambda * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        if use_whitening:
            advantages = whiten(advantages)
        return advantages.detach(), returns


    def RL_Train(self):
        self.llm_process_check_list = []
        for i in range(self.k_actions):
            self.llm_process_check_list.append(osp(osp(self.question_dir, str(i)), 'llm_done.txt'))
            # print(self.llm_process_check_list)   
        for outer_epoch in range(self.start_outer_epoch, self.args.epochs):
            log_message(f"""[Current outer_epoch]: {outer_epoch}...\n""", self.accelerator.is_main_process)
            # collect samples
            self.collect_sample()

            # export generated actions to receive the feedback
            self.export_file()

            if self.accelerator.is_main_process: 
                # let LLM know mm_cot generates all rationales
                if not os.path.exists(self.question_dir):
                    os.makedirs(self.question_dir, exist_ok=True)
                with open(osp(self.question_dir, 'mmcot_finished_signal.txt'), 'w') as f:
                    f.write('done')

            self.accelerator.wait_for_everyone()
            log_message(f"""Rationale Finished\n""", self.accelerator.is_main_process)
            log_message(f"""Waiting for the feedback \n""", self.accelerator.is_main_process)
            # wait feedback of LLM
            self.wait_feedback()
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process: 
                # let LLM know mm_cot generates all rationales and remove signal files for the next round
                for k_action in range(self.k_actions):
                    finished_signal = osp(self.question_dir, str(k_action))
                    signal_file_path_llm = osp(finished_signal, 'llm_done.txt')
                    if os.path.exists(signal_file_path_llm) and self.accelerator.is_main_process:
                        os.remove(signal_file_path_llm)

            
            self.extract_feedback()

            ppo_rl_elements_list =  list(self.ppo_rl_elements.values())
            # Assuming the list of each process can be different 
            max_length = torch.tensor(len(ppo_rl_elements_list)).to(self.device)
            # Tensor to store the maximum length across all processes
            dist.all_reduce(max_length, op=dist.ReduceOp.MAX)

            if len(ppo_rl_elements_list) < max_length.item():
                repeats = max_length.item() // len(ppo_rl_elements_list) + 1
                ppo_rl_elements_list = (ppo_rl_elements_list * repeats)[:max_length.item()]

            # Start PPO training
            for rl_step in range(self.rl_epochs):
                torch.cuda.empty_cache()
                log_message(f"""Start PPO RL Epoch: {rl_step + 1} / {self.rl_epochs} | outer_epoch: {outer_epoch + 1}\n""", self.accelerator.is_main_process)
                dataset = PPORLVisionDataset(ppo_rl_elements_list)
                sampler = PPORLBatchSampler(dataset, self.args.rl_batch_size)
                dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=lambda batch: ppo_rl_collate_fn(batch, self.tokenizer.pad_token_id))
                
                total_policy_loss, total_value_loss, total_clipfrac = [], [], []
                
                for batch_step, batch in enumerate(dataloader):
                    self.model.eval()
                    self.value_function.eval()
                    input_ids = batch['input_ids'].to(self.device)
                    image_ids = batch['image_ids'].to(self.device)
                    actions = batch['actions'].to(self.device)
                    old_logprobs = batch['logprobs'].to(self.device)
                    old_values = batch['values'].to(self.device)
                    old_rewards = batch['rewards'].to(self.device)
                    action_length = old_rewards.shape[1]
                    
                    advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, action_length, self.args.adv_normalization)
                    
                    attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.device)
                    decoder_attention_mask = actions.ne(self.tokenizer.pad_token_id).long().to(self.device)
                    decoder_attention_mask[:, 0] = 1

                    logits = self.model(
                        input_ids=input_ids,
                        image_ids=image_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=actions,
                        decoder_attention_mask=decoder_attention_mask,
                    ).logits

                    values_pred = self.value_function(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_ids=image_ids,
                        decoder_input_ids=actions,
                        decoder_attention_mask=decoder_attention_mask,
                    )

                    logprobs = logprobs_of_labels(logits[:, :-1, :], actions[:, 1:])
                    mask = actions.ne(self.tokenizer.pad_token_id).long().to(self.device)
                    logprobs = logprobs[:, :action_length]
                    values_pred = values_pred[:, :action_length]
                    mask = mask[:, 1 : action_length + 1]

                    self.model.train()
                    self.value_function.train()

                    loss, stats = self.loss(
                        logprobs=logprobs,
                        values=values_pred,
                        old_logprobs=old_logprobs,
                        advantages=advantages,
                        returns=returns,
                        mask=mask,
                    )
                    loss = loss / self.gradient_accumulation_steps
                    
                    total_policy_loss.append(stats['policy_loss'])
                    total_value_loss.append(stats['value_loss'])
                    total_clipfrac.append(stats['value_loss'])
                    
                    self.accelerator.backward(loss)
                    if (batch_step+1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                    if batch_step % self.args.log_term == 0:
                        log_message(f"""Batch PPO update: {batch_step+1} / {len(dataloader)} | outer_epoch: {outer_epoch+1}\n""", self.accelerator.is_main_process)
                        log_message(f"""policy loss: {np.mean(total_policy_loss[-4:]):.3f} / value_loss: {np.mean(total_value_loss[-4:]):.3f} / clipfrac: {np.mean(total_clipfrac[-4:]):.3f} | batch_step: {batch_step + 1} | {len(dataloader) // self.rl_batch_size + 1} / rl_step: {rl_step + 1} / {self.rl_epochs} | outer_epoch: {outer_epoch + 1}\n""", self.accelerator.is_main_process)


                # self.accelerator.wait_for_everyone()
                # if self.accelerator.is_main_process and (rl_step + 1) % 5 == 0:
                #     cur_base_dir = self.base_dir.replace('rle20', f'rle{rl_step + 1}')
                #     save_path = osp(cur_base_dir, str(outer_epoch))
                #     log_message(f"""Saving model in {save_path}\n""", self.accelerator.is_main_process)
                #     if not os.path.exists(save_path):
                #         os.makedirs(save_path, exist_ok=True)
                #     # Save the model's state_dict
                #     unwrapped_model = self.accelerator.unwrap_model(self.model)
                #     unwrapped_value_function = self.accelerator.unwrap_model(self.value_function)
                #     unwrapped_optimizer = self.accelerator.unwrap_model(self.optimizer)
                #     unwrapped_model.save_pretrained(save_path)
                #     torch.save(unwrapped_value_function.state_dict(), osp(save_path, 'value_function.pth'))
                #     torch.save(unwrapped_optimizer.state_dict(), osp(save_path, 'optimizer_state.pth'))
                #     self.tokenizer.save_pretrained(save_path)
                #     log_message(f"""Model has been saved in {save_path}\n""", self.accelerator.is_main_process)

                # self.accelerator.wait_for_everyone()
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                save_path = osp(self.base_dir, str(outer_epoch))
                log_message(f"""Saving model in {save_path}\n""", self.accelerator.is_main_process)
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                # Save the model's state_dict
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_value_function = self.accelerator.unwrap_model(self.value_function)
                unwrapped_optimizer = self.accelerator.unwrap_model(self.optimizer)
                unwrapped_model.save_pretrained(save_path)
                torch.save(unwrapped_value_function.state_dict(), osp(save_path, 'value_function.pth'))
                torch.save(unwrapped_optimizer.state_dict(), osp(save_path, 'optimizer_state.pth'))
                self.tokenizer.save_pretrained(save_path)
                log_message(f"""Model has been saved in {save_path}\n""", self.accelerator.is_main_process)

            self.accelerator.wait_for_everyone()
            log_message(f"""End outer_epoch: {outer_epoch + 1}\n""", self.accelerator.is_main_process)

        log_message(f"""Done training\n""", self.accelerator.is_main_process)
        
            
if __name__ == '__main__':
    
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

    if args.img_type is not None:
        problems, qids, name_maps, image_features = load_data_img(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems':problems, 'qids':qids, 'name_maps': name_maps, 'image_features': image_features}
    else:
        problems, qids = load_data_std(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems':problems, 'qids':qids}

    rl_t5trainer= RL_T5Trainer(
        dataframe=dataframe,
        args = args
    )

    rl_t5trainer.RL_Train()

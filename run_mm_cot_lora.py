import os
import numpy as np
import torch
import os
import re
import json
import argparse
import random
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from model import T5ForMultimodalGeneration
from utils_data import img_shape, load_data_std, load_data_std_with_correction, load_data_img, load_data_img_with_correction, ScienceQADatasetStd, ScienceQADatasetImg
from utils_prompt import *
from utils_evaluate import get_scores
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import nltk
import evaluate
import inspect
from collections.abc import Mapping
from transformers.trainer_utils import RemoveColumnsCollator
from datetime import datetime

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--model', type=str, default='declare-lab/flan-alpaca-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain', 'miniminitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    
    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet','vit'], help='type of image features')
    parser.add_argument('--train_le', type=str, default=None, help='generated rationale for the train set')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')

    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-E', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])
    parser.add_argument('--correction', type=str2bool, default=False)
    parser.add_argument('--correction_file', type=str, default='scienceqa/problems.json')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lora_r', type=int, default=32, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=64, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_bias', type=str, default='none', help='lora bias')
    parser.add_argument('--rlh', type=str, default=None, help='for RL hyperparams')
    
    

    args = parser.parse_args()
    return args
        
def T5Trainer(
    dataframe, args,
):
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True


    tokenizer = AutoTokenizer.from_pretrained(args.model)

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    problems = dataframe['problems']
    qids = dataframe['qids']
    train_qids = qids['train']
    test_qids = qids['test']
    val_qids = qids['val']
    gpu_count = torch.cuda.device_count()

    model_id = f'mm_cot_lora_r{args.lora_r}_a{args.lora_alpha}_d{args.lora_dropout}_s{args.seed}_lr{args.lr}_bs{args.bs * gpu_count}_e{args.epoch}_len{args.output_len}'
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:    
        save_dir = f'{args.model}/{model_id}'

    if args.img_type is not None:
        patch_size = img_shape[args.img_type]
        model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size) 
        name_maps = dataframe['name_maps'] 
        image_features = dataframe['image_features']
        train_set = ScienceQADatasetImg(
            problems,
            train_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.train_le,
        )
        eval_set = ScienceQADatasetImg(
            problems,
            val_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.eval_le,
        )
        test_set = ScienceQADatasetImg(
            problems,
            test_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.test_le,
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model) 
        train_set = ScienceQADatasetStd(
            problems,
            train_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.train_le,
        )
        eval_set = ScienceQADatasetStd(
            problems,
            val_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.eval_le,
        )
        test_set = ScienceQADatasetStd(
            problems,
            test_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.test_le,
        )

    datacollator = DataCollatorForSeq2Seq(tokenizer)
    console.log(f"""[Model_Parameters]: {model.num_parameters()}\n""")


    def extract_ans(ans):
        pattern = re.compile(r'The answer is \(([A-Z])\)')
        res = pattern.findall(ans)
        
        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED" 
        return answer  

    # accuracy for answer inference
    def compute_metrics_acc(eval_preds):
        if args.use_generate:
            preds = np.where(eval_preds.predictions != -100, eval_preds.predictions, tokenizer.pad_token_id)
            targets = eval_preds.label_ids
            #preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option:
                correct +=1 
        return {'accuracy': 1.0*correct/len(targets)}
    
    # rougel for rationale generation
    metric = evaluate.load("rouge")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics_rougel(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(preds, targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    
    # only use the last model for evaluation to save time
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            report_to="none",
            remove_unused_columns=False,
        )
    # evaluate at each epoch
    else:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="accuracy" if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else "rougeL",
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            load_best_model_at_end=True,
            report_to="none",
            remove_unused_columns=False,
        )

    # from https://github.com/haotian-liu/LLaVA
    def find_all_linear_names(model):
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        # if 'lm_head' in lora_module_names: # needed for 16-bit
        #     lora_module_names.remove('lm_head')
        return list(lora_module_names)


    if args.evaluate_dir is None:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=find_all_linear_names(model), inference_mode=False, 
            r=args.lora_r, lora_alpha=args.lora_alpha, bias=args.lora_bias, lora_dropout=args.lora_dropout) 
        
        model = get_peft_model(model, peft_config)
        model.save_pretrained(save_dir)
    else:
        config = PeftConfig.from_pretrained(save_dir,local_files_only=True)
        model = PeftModel.from_pretrained(model, save_dir,local_files_only=True)
        # 학습 가능한 파라미터 출력 (LoRA 적용 후)
        print("After applying LoRA:")
        model.print_trainable_parameters()
        # 학습 가능한 파라미터 수 계산 (LoRA 적용 후)
        total_trainable_params_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters in the LoRA model: {total_trainable_params_lora}")


    model.print_trainable_parameters()
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters in the original model: {total_trainable_params}")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics_acc if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else compute_metrics_rougel
    )

    if args.evaluate_dir is None:
        trainer.train()
        model.save_pretrained(save_dir)
    

    metrics = trainer.evaluate(eval_dataset = test_set, max_length=args.output_len)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    predict_results = trainer.predict(test_dataset=test_set, max_length=args.output_len) 
    if trainer.is_world_process_zero():
        if args.use_generate:
            preds = np.where(predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id)
            targets = predict_results.label_ids
        else:
            preds = predict_results.predictions[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            targets = predict_results.label_ids
            preds = preds.argmax(axis=2)

        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        targets = tokenizer.batch_decode(
            targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        results_ans = {}
        results_rationale = {}
        results_reference = {}
        
        num_fail = 0
        for idx, qid in enumerate(test_qids):
            pred = preds[int(idx)]
            ref = targets[int(idx)]
            extract_pred = extract_ans(pred)
            if extract_pred != "FAILED":
                if extract_pred in args.options:
                    extract_pred = args.options.index(extract_pred)
                else:
                    extract_pred = random.choice(range(0,len(args.options)))
            else:
                num_fail += 1
                extract_pred = random.choice(range(len(args.options))) # random choose one option
            results_ans[str(qid)] = extract_pred
            results_rationale[str(qid)] = pred
            results_reference[str(qid)] = ref
        if args.correction == True:
            scores = get_scores(results_ans, results_rationale, results_reference, os.path.join(args.data_root, args.correction_file)) # scienceqa/2e5_false.json
        else:
            scores = get_scores(results_ans, results_rationale, results_reference, os.path.join(args.data_root, "scienceqa/problems.json"))
        preds = [pred.strip() for pred in preds]
        output_data = {
                "num_fail": num_fail,
                "scores": scores,
                "preds": preds,
                 "labels": targets}
        output_prediction_file = os.path.join(save_dir,"predictions_ans_test.json")
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(output_data, indent=4))
    
    # generate the rationale for the eval set
    if args.prompt_format == "QCM-LE" or args.prompt_format == "QCM-E":
        torch.cuda.empty_cache()
        del predict_results, preds, targets
        predict_results = trainer.predict(test_dataset=eval_set, max_length=args.output_len) 
        if trainer.is_world_process_zero():
            if args.use_generate:
                preds = np.where(predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id)
                targets = predict_results.label_ids
            else:
                preds = predict_results.predictions[0]
                preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                targets = predict_results.label_ids
                preds = preds.argmax(axis=2)

            preds = tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            targets = tokenizer.batch_decode(
                targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            preds = [pred.strip() for pred in preds]
            output_data = {"preds": preds,
                 "labels": targets}
            output_prediction_file = os.path.join(save_dir,"predictions_ans_eval.json")
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output_data, indent=4))
    
    # for train_set
    current_time = datetime.now().strftime("%H:%M:%S")
    console.log(f"""[{current_time}] Train dataset Evaluation Start\n####""")
    if args.prompt_format == "QCM-LE" or args.prompt_format == "QCM-E":
        torch.cuda.empty_cache()
        del predict_results, preds, targets
        predict_results = trainer.predict(test_dataset=train_set, max_length=args.output_len)
        if trainer.is_world_process_zero():
            if args.use_generate:
                preds = np.where(predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id)
                targets = predict_results.label_ids
            else:
                preds = predict_results.predictions[0]
                preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                targets = predict_results.label_ids
                preds = preds.argmax(axis=2)

            preds = tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            targets = tokenizer.batch_decode(
                targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            preds = [pred.strip() for pred in preds]
            output_data = {"preds": preds,
                 "labels": targets}
            output_prediction_file = os.path.join(save_dir,"predictions_ans_train.json")
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output_data, indent=4))

    

if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    

    if args.img_type is not None:
        if args.correction == True:
            problems, qids, name_maps, image_features = load_data_img_with_correction(args)  # probelms, test question ids, shot example ids
            dataframe = {'problems':problems, 'qids':qids, 'name_maps': name_maps, 'image_features': image_features}
        else:
            problems, qids, name_maps, image_features = load_data_img(args)  # probelms, test question ids, shot example ids
            dataframe = {'problems':problems, 'qids':qids, 'name_maps': name_maps, 'image_features': image_features}
    else:
        if args.correction == True:
            problems, qids = load_data_std_with_correction(args)  # probelms, test question ids, shot example ids
            dataframe = {'problems':problems, 'qids':qids}
        else:
            problems, qids = load_data_std(args)  # probelms, test question ids, shot example ids
            dataframe = {'problems':problems, 'qids':qids}

    T5Trainer(
        dataframe=dataframe,
        args = args
    )

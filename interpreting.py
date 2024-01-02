from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from datasets import load_dataset
from interpreter_wrapper import InterpreterWrapper
import torch
from dataclasses import dataclass, field
from sacrebleu import corpus_bleu
# logging.set_verbosity_info()
from transformers.utils import add_start_docstrings
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser
from interpreter_wrapper import InterpreterWrapper
# import argparse
import tqdm
import os
import seaborn as sns
from interpreting_methods import *
import json
import torch.nn.functional as F
import torch.nn as nn

import matplotlib    
print(matplotlib.matplotlib_fname())
# plt.rc("font",family="AR PL UKai CN") ###修改了这一行
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

@dataclass
class InferenceArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    src_lang: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    tgt_lang: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_length: int = field(
        default=512, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    use_cache: bool = field(
        default=False, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    out_file: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    bs: int = field(
        default=32, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    beam: int = field(
        default=4, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    early_stopping: bool = field(
        default=True, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    do_sample: bool = field(
        default=True, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    nmt_checkpoint: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    sinmt_checkpoint: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    model_type: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )



# xxx: 2023-04-11, customized args
@dataclass
@add_start_docstrings(Seq2SeqTrainingArguments.__doc__)
class InterpreterSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    ngram: int = field(
        default=3,
        metadata={"help": "ngram for interpretability loss"},
    )
    stop_attn_grad: bool = field(
        default=False,
        metadata={"help": "Stop gradient from attn"},
    )
    stop_x_grad: bool = field(
        default=False,
        metadata={"help": "Stop gradient from x_out"},
    )
    interpretability_lambda: float = field(
        default=0.1, metadata={"help": "weight for the interpretability loss"}
    )
    interpretability_loss_type: str = field(
        default="kl", metadata={"help": "interpretability loss type"}
    )
    output_embed_dim: int = field(
        default=1024, metadata={"help": "output embedding dimension"}
    )
    full_context_interpreter: bool = field(
        default=False, metadata={"help": "full context interpret"}
    )
    nmt_model_checkpoint: str = field(
        default="google/mt5-large", metadata={"help": "nmt model checkpoint"}
    )
    full_model_checkpoint: str = field(
        default="", metadata={"help": "full model checkpoint"}
    )





lang_dict = {"en": "English", "de": "German", "zh": "Chinese"}

def plot(src_str_list, pred_str_list, attention, prefix="", save_dir="."):
    # plot heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(attention, xticklabels=pred_str_list, yticklabels=src_str_list, cmap="YlGnBu")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Source")
    plt.savefig("{}/attention_{}.png".format(save_dir, prefix))

def main():

    lang_dict = {"en": "English", "de": "German", "zh": "Chinese"}
    parser = HfArgumentParser((InterpreterSeq2SeqTrainingArguments, InferenceArguments))
    training_args, args = parser.parse_args_into_dataclasses()

    batch_size = args.bs
    beam_size = args.beam
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    data_dir = args.data_path
    dataset_name = args.dataset_name
    output_dir = training_args.output_dir
    model_type = args.model_type

    prefix = f"translate {lang_dict[src_lang]} to {lang_dict[tgt_lang]}: "

    train_file = f'{data_dir}/train_{dataset_name}.json'
    test_file = f'{data_dir}/dev_{dataset_name}.json'
    nmt_checkpoint = args.nmt_checkpoint
    sinmt_checkpoint = args.sinmt_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    data = load_dataset("json", data_files={"train":train_file, "test":test_file}, field="data")

    prefix = f"translate {lang_dict[src_lang]} to {lang_dict[tgt_lang]}: "

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(nmt_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(nmt_checkpoint)
    
    # check the targe model
    print(sinmt_checkpoint)
    if sinmt_checkpoint is not None:
        print(f"load {os.path.join(sinmt_checkpoint, 'pytorch_model.bin')} as the target model")
        model = InterpreterWrapper(model, tokenizer, training_args)
        model.load_state_dict(torch.load(os.path.join(sinmt_checkpoint, 'pytorch_model.bin')))
    else:
        print(f"load {nmt_checkpoint} as the target model")

    model.eval()
    model.to(device)

    sample_num = 100000
    
    # for data_key in ["test", "train"]:
    for data_key in ["train"]:
        srcs = data[data_key][src_lang]
        tgts = data[data_key][tgt_lang]

        sample_window = int(len(srcs) / sample_num) + 1

        srcs = srcs[::sample_window]
        tgts = tgts[::sample_window]

        # f = open(f"{output_dir}/{model_type}_{dataset_name}_{data_key}.pt", "w")
        method="pd"
        fpath = f"{output_dir}/{method}_{model_type}_{dataset_name}_{src_lang}{tgt_lang}_{data_key}.pt"
        all_evidence_words = []
        all_pred_words = []
        for i in tqdm.trange(0, len(srcs), batch_size):
            src = srcs[i:i+batch_size]
            tgt = tgts[i:i+batch_size]

            inputs = tokenizer([prefix + item for item in src], text_target=tgt, return_tensors="pt", padding=True)
            # inputs = tokenizer(src, text_target=tgt, return_tensors="pt", padding=True)
            inputs.to(device)

            src_lens = inputs["attention_mask"].sum(dim=1)
            tgt_lens = (inputs["labels"] != tokenizer.pad_token_id).sum(dim=1)
            src_mask = inputs["attention_mask"]
            tgt_mask = inputs["labels"] != tokenizer.pad_token_id
            ctx_mask = torch.cat((src_mask, tgt_mask), dim=1)

            sample = dict()
            sample["net_input"] = inputs
            # evidence_tokens_list, preds_list = attention(model, sample)
            # evidence_tokens_list, preds_list = gradient(model, sample)
            evidence_tokens_list, preds_list, scores = pd(model, sample, fc=True)

            fig, ax = plt.subplots(figsize=(10, 10))
            # print(scores.size())
            # print(src_lens[0]+tgt_lens[0])
            # print(src_lens)
            # print(tgt_lens)
            print(scores[0].size())
            score_0 = scores[0][:tgt_lens[0], :].masked_select(ctx_mask[0].unsqueeze(0).repeat(tgt_lens[0], 1).bool())
            score_0 = score_0.view(tgt_lens[0], src_lens[0]+tgt_lens[0])
            ax = sns.heatmap(np.around(score_0.cpu().numpy(), 3), 
                                        xticklabels=tokenizer.convert_ids_to_tokens(list(inputs["input_ids"][0][:src_lens[0]]) + list(model._shift_right(inputs["labels"])[0][:tgt_lens[0]])), 
                                        yticklabels=tokenizer.convert_ids_to_tokens(inputs["labels"][0][:tgt_lens[0]]), 
                                        annot=True, cmap="YlGnBu")
            # ax = sns.heatmap(scores[0].cpu().numpy(), cmap="YlGnBu")
            # ax = sns.heatmap(scores[0].cpu().numpy()[:tgt_lens[0], :src_lens[0]], cmap="YlGnBu")
            ax.set_xlabel("Source")
            ax.set_ylabel("Predicted")
            plt.savefig("test.png")

            # print(scores[0].cpu().numpy())

            # for evidence_token, preds in zip(evidence_tokens_list, preds_list):
            #     evi = tokenizer.convert_ids_to_tokens(evidence_token)
            #     pred = tokenizer.convert_ids_to_tokens([preds])
            #     print(f"{evi} {pred}")
            exit()
            # print(evidence_str_list)
            # print(preds_str_list)
            all_evidence_words += evidence_tokens_list
            all_pred_words += preds_list

            if i % 1000 == 0:
                torch.save({"evidence": all_evidence_words, "pred":all_pred_words}, fpath)
        torch.save({"evidence": all_evidence_words, "pred":all_pred_words}, fpath)


if __name__ == "__main__":
    main()
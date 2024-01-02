import torch
# from fairseq.models import SaliencyManager
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns
import torch.nn.functional as F
import torch.nn as nn

def posterior(model, sample, fc, topk=1):
    
    with torch.no_grad():
        model.eval()

        src_tokens = sample["net_input"]["src_tokens"]
        tgt_tokens = sample["net_input"]["prev_output_tokens"]

        net_output = model(**sample["net_input"], full_context_interpreter=True) # bxTxV
        del sample
        outputs = net_output[0]
        p_z2y = net_output[-1]["z2y"] # bx(S+T)xV
        preds = torch.argmax(outputs, dim=-1) # b x T
        p_src2z = net_output[-1]["attn"][0].to(p_z2y) # bxTxS
        p_tgt2z = net_output[-1]["self_attn"][0] # bxTxT 
        del net_output

        # get final contribution score
        p_z2pred = torch.gather(p_z2y, dim=2, index=preds.unsqueeze(1).repeat(1, p_z2y.size(1), 1)) # b x (S+T) x T
        p_c2z = torch.cat((p_src2z, p_tgt2z), dim=-1) # bxTx(S+T)
        p_c2pred = p_c2z * p_z2pred.transpose(1, 2) # bxTx(S+T)

        # for src tokens
        p_src2pred = p_c2pred[:, :, :p_src2z.size(-1)]
        src_evidence_index = torch.topk(p_src2pred, topk, dim=-1)[1] # select top k src tokens [pad tokens' weights are zero]
        repeat_src_tokens = src_tokens.unsqueeze(1).repeat(1, src_evidence_index.size(1), 1)
        src_evidence_tokens = torch.gather(repeat_src_tokens, dim=-1, index=src_evidence_index) # bs x T x K

        preds = torch.argmax(outputs, dim=-1)
        valid_src_evidence_tokens = src_evidence_tokens[tgt_tokens != 1] # remove invalid target tokens [pad]
        valid_preds = preds[tgt_tokens != 1] # remove invalid target tokens [pad]

        if fc:
            # for tgt tokens
            p_tgt2pred = p_c2pred[:, :, p_src2z.size(-1):]
            tgt_evidence_index = torch.topk(p_tgt2pred, topk, dim=-1)[1] # select top k src tokens [pad tokens' weights are zero]
            repeat_tgt_tokens = tgt_tokens.unsqueeze(1).repeat(1, tgt_evidence_index.size(1), 1)
            tgt_evidence_tokens = torch.gather(repeat_tgt_tokens, dim=-1, index=tgt_evidence_index) # bs x T x K

            valid_tgt_evidence_tokens = tgt_evidence_tokens[tgt_tokens != 1] # remove invalid target tokens [pad]

            evidence_tokens_list = [s+t for s, t in zip(valid_src_evidence_tokens.tolist(), valid_tgt_evidence_tokens.tolist())]
            preds_list = valid_preds.tolist()
        else:
            evidence_tokens_list = [s for s in valid_src_evidence_tokens.tolist()]
            preds_list = valid_preds.tolist()

        return evidence_tokens_list, preds_list


def prior(model, sample, fc, topk=1):
    # attention (transformer), prior (sinmt), attention (align) 
    with torch.no_grad():
        model.eval()

        src_tokens = sample["net_input"]["src_tokens"]
        tgt_tokens = sample["net_input"]["prev_output_tokens"]

        net_output = model(**sample["net_input"])

        del sample
        outputs = net_output[0] # bxTxV
        p_src2z = net_output[-1]["attn"][0] # bxTxS
        p_tgt2z = net_output[-1]["self_attn"][0]
        del net_output

        # for src tokens
        src_evidence_index = torch.topk(p_src2z, topk, dim=-1)[1] # select top k src tokens [pad tokens' weights are zero]
        repeat_src_tokens = src_tokens.unsqueeze(1).repeat(1, src_evidence_index.size(1), 1)
        src_evidence_tokens = torch.gather(repeat_src_tokens, dim=-1, index=src_evidence_index) # bs x T x K

        preds = torch.argmax(outputs, dim=-1)
        valid_src_evidence_tokens = src_evidence_tokens[tgt_tokens != 1] # remove invalid target tokens [pad]
        valid_preds = preds[tgt_tokens != 1] # remove invalid target tokens [pad]

        if fc:
            # for tgt tokens
            tgt_evidence_index = torch.topk(p_tgt2z, topk, dim=-1)[1]
            repeat_tgt_tokens = tgt_tokens.unsqueeze(1).repeat(1, tgt_evidence_index.size(1), 1)
            tgt_evidence_tokens = torch.gather(repeat_tgt_tokens, dim=-1, index=tgt_evidence_index)

            valid_tgt_evidence_tokens = tgt_evidence_tokens[tgt_tokens != 1]

            evidence_tokens_list = [s+t for s, t in zip(valid_src_evidence_tokens.tolist(), valid_tgt_evidence_tokens.tolist())]
            preds_list = valid_preds.tolist()
        else:
            evidence_tokens_list = [s for s in valid_src_evidence_tokens.tolist()]
            preds_list = valid_preds.tolist()

        return evidence_tokens_list, preds_list, p_src2pred


def attention(model, sample, fc=False, align_layer=5, topk=1):
    # attention (transformer), prior (sinmt), attention (align) 
    with torch.no_grad():
        model.eval()

        src_tokens = sample["net_input"].input_ids
        tgt_tokens = sample["net_input"].labels

        net_output = model(**sample["net_input"], output_attentions=True)

        del sample
        # print(net_output.keys())
        outputs = net_output.logits # bxTxV
        p_src2z = net_output.cross_attentions[-2].mean(dim=1) # bxTxS
        p_tgt2z = net_output.decoder_attentions[-1].mean(dim=1)

        # clip attentions
        src_eos_index = (src_tokens == 1)
        src_eos_repeat_index = src_eos_index.unsqueeze(1).repeat(1, p_src2z.size(1), 1)
        p_src2z[:, :,  :6] = 0 # Dont't count instruction words
        p_src2z[src_eos_repeat_index] = 0

        del net_output

        # for src tokens
        src_evidence_index = torch.topk(p_src2z, topk, dim=-1)[1] # select top k src tokens [pad tokens' weights are zero]
        repeat_src_tokens = src_tokens.unsqueeze(1).repeat(1, src_evidence_index.size(1), 1)
        src_evidence_tokens = torch.gather(repeat_src_tokens, dim=-1, index=src_evidence_index) # bs x T x K

        preds = torch.argmax(outputs, dim=-1)
        valid_src_evidence_tokens = src_evidence_tokens[tgt_tokens != 0] # remove invalid target tokens [pad]
        valid_preds = preds[tgt_tokens != 0] # remove invalid target tokens [pad]

        if fc:
            # for tgt tokens
            tgt_evidence_index = torch.topk(p_tgt2z, topk, dim=-1)[1]
            repeat_tgt_tokens = tgt_tokens.unsqueeze(1).repeat(1, tgt_evidence_index.size(1), 1)
            tgt_evidence_tokens = torch.gather(repeat_tgt_tokens, dim=-1, index=tgt_evidence_index)

            valid_tgt_evidence_tokens = tgt_evidence_tokens[tgt_tokens != 1]

            evidence_tokens_list = [s+t for s, t in zip(valid_src_evidence_tokens.tolist(), valid_tgt_evidence_tokens.tolist())]
            preds_list = valid_preds.tolist()
        else:
            evidence_tokens_list = [s for s in valid_src_evidence_tokens.tolist()]
            preds_list = valid_preds.tolist()

        return evidence_tokens_list, preds_list


def gradient(model, sample, fc=False, topk=1):

    model.eval()
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    src_tokens = sample["net_input"].input_ids
    tgt_tokens = sample["net_input"].labels

    inputs = sample["net_input"]

    net_output = model(**sample["net_input"])

    vocab_size = model.shared.weight.size(0)

    token_ids = inputs.input_ids
    token_ids_tensor_one_hot = F.one_hot(token_ids, vocab_size).to(model.shared.weight)
    token_ids_tensor_one_hot.requires_grad = True

    inputs_embeds = torch.matmul(token_ids_tensor_one_hot, model.shared.weight.unsqueeze(0))
    inputs["inputs_embeds"] = inputs_embeds
    del inputs["input_ids"]

    # sample = dict()
    # sample["net_input"] = inputs

    outputs = model(**inputs)
    logits = outputs.logits

    preds = torch.argmax(logits, dim=-1)
    loss = loss_fct(logits.view(-1, vocab_size), preds.view(-1))
    pred_p = torch.exp(-loss.view(len(logits), -1))
    p_src2pred = []
    for i in range(pred_p.size(1)):
        sum_p = sum(pred_p[:, i])
        gradients = torch.autograd.grad(sum_p, token_ids_tensor_one_hot, retain_graph=True)
        norm_gradient = torch.norm(gradients[0], dim=2)
        scores = norm_gradient / (torch.sum(norm_gradient, dim=1).view(-1, 1) + 1e-8)
        p_src2pred.append(scores.unsqueeze(1))
        model.zero_grad()
    p_src2pred = torch.cat(p_src2pred, dim=1)

    # for src tokens
    src_evidence_index = torch.topk(p_src2pred, topk, dim=-1)[1] # select top k src tokens [pad tokens' weights are zero]
    repeat_src_tokens = src_tokens.unsqueeze(1).repeat(1, src_evidence_index.size(1), 1)
    src_evidence_tokens = torch.gather(repeat_src_tokens, dim=-1, index=src_evidence_index) # bs x T x K

    valid_src_evidence_tokens = src_evidence_tokens[tgt_tokens != 0] # remove invalid target tokens [pad]
    valid_preds = preds[tgt_tokens != 0] # remove invalid target tokens [pad]

    if fc:
        # for tgt tokens
        p_tgt2pred = torch.cat(p_tgt2pred, dim=1).to(p_outputs)
        p_c2pred = torch.cat((p_src2pred, p_tgt2pred), dim=-1)

        ctx_evidence_index = torch.topk(p_c2pred, topk, dim=-1)[1]
        ctx_tokens = torch.cat((src_tokens, tgt_tokens), dim=1)

        repeat_ctx_tokens = ctx_tokens.unsqueeze(1).repeat(1, ctx_evidence_index.size(1), 1)
        ctx_evidence_tokens = torch.gather(repeat_ctx_tokens, dim=-1, index=ctx_evidence_index)

        valid_ctx_evidence_tokens = ctx_evidence_tokens[tgt_tokens != 1]

        evidence_tokens_list = [s for s in valid_ctx_evidence_tokens.tolist()]
        preds_list = valid_preds.tolist()
    else:
        evidence_tokens_list = [s for s in valid_src_evidence_tokens.tolist()]
        preds_list = valid_preds.tolist()

    return evidence_tokens_list, preds_list
            

def pd(model, sample, fc=False, topk=1, pad=0):

    with torch.no_grad():
        model.eval()
        pd_matrix = [] # src x b x T

        # 1-get explained objects
        inputs = sample["net_input"]
        outputs = model(**inputs)
        p_outputs = torch.softmax(outputs.logits, dim=-1)
        origin_preds = torch.argmax(outputs.logits, dim=-1)
        origin_pred_p = torch.max(p_outputs, dim=-1)[0].view(p_outputs.size(0), p_outputs.size(1), 1) # original predictions's probability

        # 2-get attributed objects
        src_tokens = sample["net_input"].input_ids
        labels = sample["net_input"].labels
        decoder_input_ids = model._shift_right(labels)
        src_mask = (src_tokens != pad)
        neg_src_mask = ~src_mask * -10 # pad token's mask is -10

        if fc:
            ctx_len = src_tokens.size(-1) + decoder_input_ids.size(-1)
            ctx_tokens = torch.cat((src_tokens, decoder_input_ids), dim=1)
            tgt_mask = (labels != pad)
            neg_tgt_mask = ~tgt_mask * -10 # pad token's mask is -10

            neg_ctx_mask = torch.cat((neg_src_mask, neg_tgt_mask), dim=-1)
            neg_ctx_mask = neg_ctx_mask.unsqueeze(1).to(p_outputs)
        else:
            ctx_len = src_tokens.size(-1)
            ctx_tokens = src_tokens
            neg_ctx_mask = neg_src_mask
            neg_ctx_mask = neg_ctx_mask.unsqueeze(1).to(p_outputs)

        vocab_size = model.shared.weight.size(0)
        inputs_embeds = torch.matmul(F.one_hot(src_tokens, vocab_size).to(model.shared.weight), model.shared.weight.unsqueeze(0))
        decoder_inputs_embeds = torch.matmul(F.one_hot(decoder_input_ids, vocab_size).to(model.shared.weight), model.shared.weight.unsqueeze(0))
        inputs["inputs_embeds"] = inputs_embeds
        inputs["decoder_inputs_embeds"] = decoder_inputs_embeds
        inputs["decoder_attention_mask"] = tgt_mask

        for zero_pos in range(ctx_len):
            offset = zero_pos - src_tokens.size(-1)
            src_zero_pos, tgt_zero_pos = (zero_pos, None) if offset < 0 else (None, offset)
            
            zero_inputs_mask = torch.ones(src_tokens.size()).to(inputs["inputs_embeds"])
            if src_zero_pos is not None:
                zero_inputs_mask[:, src_zero_pos] = 0
                # print("src zero")
            zero_decoder_inputs_mask = torch.ones(decoder_input_ids.size()).to(inputs["decoder_inputs_embeds"])
            if tgt_zero_pos is not None:
                zero_decoder_inputs_mask[:, tgt_zero_pos] = 0
                # print("tgt zero: ", tgt_zero_pos)
            # print(zero_decoder_inputs_mask)

            zero_inputs_embeds = inputs["inputs_embeds"] * zero_inputs_mask.unsqueeze(-1)
            zero_decoder_inputs_embeds = inputs["decoder_inputs_embeds"] * zero_decoder_inputs_mask.unsqueeze(-1)
            
            zero_inputs = {
                "inputs_embeds": zero_inputs_embeds,
                "attention_mask": inputs["attention_mask"],
                "decoder_inputs_embeds": zero_decoder_inputs_embeds,
                # "decoder_attention_mask": inputs["decoder_attention_mask"],
                "labels": inputs["labels"]
            }

            outputs = model(**zero_inputs)
            p_outputs = torch.softmax(outputs.logits, dim=-1)
            pred_p = torch.gather(p_outputs, dim=-1, index=origin_preds.unsqueeze(2)).view(p_outputs.size(0), p_outputs.size(1), 1) # bxTx1
            # print(pred_p.squeeze()[0, :12])
            delta_pred_p = origin_pred_p - pred_p
            pd_matrix.append(delta_pred_p)
            # print(delta_pred_p.squeeze().mean(dim=-1))

        p_c2pred = torch.cat(pd_matrix, dim=-1) # bxTx(S+T) / # bxTxS
        # print(tgt_tokens)
        # set scores of pad tokens in context as -10
        p_c2pred += neg_ctx_mask # bs x 1 x (S+T)/S
        p_c2pred[:, :,  :6] = -10 # ignore the prefix tokens
        p_c2pred[labels == pad, :] = -10 # set pad tgt tokens's weight to 0
        p_c2pred = torch.softmax(p_c2pred, dim=-1)

        ctx_evidence_index = torch.topk(p_c2pred, topk, dim=-1)[1]

        repeat_ctx_tokens = ctx_tokens.unsqueeze(1).repeat(1, ctx_evidence_index.size(1), 1)
        ctx_evidence_tokens = torch.gather(repeat_ctx_tokens, dim=-1, index=ctx_evidence_index)

        valid_ctx_evidence_tokens = ctx_evidence_tokens[decoder_input_ids != pad]
        valid_preds = origin_preds[decoder_input_ids != pad] # remove invalid target tokens [pad]

        evidence_tokens_list = [s for s in valid_ctx_evidence_tokens.tolist()]
        preds_list = valid_preds.tolist()

        return evidence_tokens_list, preds_list, p_c2pred



def alti_plus2(model, sample, fc, topk=1):
    pass





def posterior_d(generator, model, sample, logger, fc, topk=1):
    
    with torch.no_grad():
        model.eval()

        src_tokens = sample["net_input"]["src_tokens"]
        # tgt_tokens = sample["net_input"]["prev_output_tokens"]

        # net_output = model(**sample["net_input"])
        # print(net_output.keys())

        net_output = generator(sample) # bxTxV
        max_len = 0
        new_prev_output_tokens = torch.ones(src_tokens.size(0), 256)
        new_prev_output_tokens[:, 0] = 2
        for sent_id, gen in enumerate(net_output):
            model_output = gen[0]
            tokens = model_output["tokens"]
            new_prev_output_tokens[sent_id][1:len(tokens)+1] = tokens
            if len(tokens) > max_len:
                max_len = len(tokens)

        new_prev_output_tokens = new_prev_output_tokens[:, :max_len].to(src_tokens)
        sample["net_input"]["prev_output_tokens"] = new_prev_output_tokens
        tgt_tokens = new_prev_output_tokens
        net_output = model(**sample["net_input"], full_context_interpreter=True)

        # del sample
        outputs = net_output[0]
        p_z2y = net_output[-1]["z2y"] # bx(S+T)xV
        preds = torch.argmax(outputs, dim=-1) # b x T
        p_src2z = net_output[-1]["attn"][0].to(p_z2y) # bxTxS
        p_tgt2z = net_output[-1]["self_attn"][0] # bxTxT 
        del net_output

        # get final contribution score
        p_z2pred = torch.gather(p_z2y, dim=2, index=preds.unsqueeze(1).repeat(1, p_z2y.size(1), 1)) # b x (S+T) x T
        p_c2z = torch.cat((p_src2z, p_tgt2z), dim=-1) # bxTx(S+T)
        p_c2pred = p_c2z * p_z2pred.transpose(1, 2) # bxTx(S+T)

        # for src tokens
        p_src2pred = p_c2pred[:, :, :p_src2z.size(-1)]
        return p_src2pred
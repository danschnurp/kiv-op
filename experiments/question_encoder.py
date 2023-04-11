import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import LongformerConfig, LongformerModel


def prepare_tok_model():
    # Loading the tokenizer from the pretrained model.
    tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")
    # tokenizer eos_token -> sep , bos_token -> cls
    tokenizer.bos_token = "cls"
    tokenizer.eos_token = "sep"
    # Loading the pretrained model.
    model = LongformerModel.from_pretrained("UWB-AIR/MQDD-duplicates")
    return tokenizer, model


def encode_question(question, tokenizer, model):
    max_length = 512

    # Encoding the question into a list of integers.
    encoded_question = tokenizer.encode(
        question,
        max_length=max_length,
        truncation=True)

    # Preparing the encoded question for the model.
    encoded_question = tokenizer.prepare_for_model(
        encoded_question,
        [],
        max_length=max_length,
        padding="max_length",
        return_token_type_ids=True,
        truncation=True, return_tensors="pt")

    encoded_question.data["input_ids"] = torch.reshape(encoded_question.data["input_ids"],
                                                       (1, encoded_question.data["input_ids"].shape[0]))
    encoded_question.data["token_type_ids"] = torch.reshape(encoded_question.data["token_type_ids"],
                                                            (1, encoded_question.data["token_type_ids"].shape[0]))
    encoded_question.data["attention_mask"] = torch.reshape(encoded_question.data["attention_mask"],
                                                            (1, encoded_question.data["attention_mask"].shape[0]))
    _, encoded_question_result = model.__call__(**encoded_question)

    return np.squeeze(encoded_question_result.detach().numpy())
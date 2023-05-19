from torch import cuda, version, backends
from tqdm import tqdm

import numpy as np
import torch

from web.search.brain.neural_loader import load_nett


def control_torch():
    """
    test if cuda is available
    """
    print("cuda availability: " + str(cuda.is_available()))

    import gc
    gc.collect()

    if not cuda.is_available():
        return "cpu"

    print("version: " + version.cuda)
    cuda.empty_cache()
    # Storing ID of current CUDA device
    cuda_id = cuda.current_device()
    print(f"ID of current CUDA device:{cuda.current_device()}")
    print(f"Name of current CUDA device:{cuda.get_device_name(cuda_id)}")

    # VERY IMPORTANT
    ######################################
    backends.cudnn.enabled = True  #
    backends.cudnn.benchmark = True  #
    backends.cudnn.deterministic = True  #
    #######################################
    return "cuda"


def prepare_tok_model():
    model, tokenizer, position_ids = load_nett(
        model_location="C:/Users/dartixus/PycharmProjects/kiv-op/web/search/brain/model.pt")
    max_len = int(len(torch.squeeze(position_ids)) / 2.)
    this_device = control_torch()
    model.to(this_device)
    return tokenizer, model, this_device, max_len


def tokenize_question(question, tokenizer, max_length):
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
        return_tensors="pt")

    encoded_question.data["input_ids"] = torch.reshape(encoded_question.data["input_ids"],
                                                       (1,
                                                        encoded_question.data["input_ids"].shape[
                                                            0]))
    encoded_question.data["attention_mask"] = torch.reshape(
        encoded_question.data["attention_mask"],
        (1, encoded_question.data["attention_mask"].shape[0]))
    encoded_question.data["token_type_ids"] = torch.reshape(
        encoded_question.data["token_type_ids"],
        (1, encoded_question.data["token_type_ids"].shape[0]))

    return encoded_question


def encode_questions(questions, tokenizer, model, this_device, max_length, batch_size=1, ):
    """
    This function encodes a given question using a tokenizer and a pre-trained language model, with a specified maximum
    length.
    """
    print("encoding...")
    encoded_result_list = []
    for q in tqdm(questions):
        tokenized_question = tokenize_question(q, tokenizer, max_length)
        tokenized_question.to(this_device)
        # calling the pre-trained language model UWB-AIR/MQDD-duplicates
        encoded_question_result = model.encode_only(tokenized_question.data)
        # unpacking results from batch size to list
        encoded_result_list.extend(encoded_question_result.detach().cpu().numpy().tolist())

    return np.array(encoded_result_list)

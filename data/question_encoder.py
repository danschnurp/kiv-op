from torch import cuda, version, device, backends
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import LongformerModel


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


def prepare_tok_model(max_length=512):
    # Loading the tokenizer from the pretrained model.
    tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")
    # tokenizer eos_token -> sep , bos_token -> cls
    tokenizer.bos_token = "cls"
    tokenizer.eos_token = "sep"
    tokenizer.init_kwargs["bos_token"] = "cls"
    tokenizer.init_kwargs["eos_token"] = "sep"
    # Loading the pretrained model.
    model = LongformerModel.from_pretrained("UWB-AIR/MQDD-duplicates")

    # creating example input
    tokenized_question_example = tokenizer(
        "sample question", max_length=max_length,
        padding="max_length",
        return_token_type_ids=True,
        truncation=True, return_tensors="pt")

    this_device = "cpu"
    model.to(this_device)
    tokenized_question_example.to(this_device)

    return tokenizer, model, tokenized_question_example, this_device


def encode_question(question, tokenizer, model, max_length=512, this_device="cpu"):
    """
    This function encodes a given question using a tokenizer and a pre-trained language model, with a specified maximum
    length.

    :param question: The input question that needs to be encoded
    :param tokenizer: The tokenizer is a tool used to convert text into numerical values that can be processed by model.
    :param model: The model parameter refers to the pre-trained language model.
    :param max_length: The maximum length of the input sequence that the model can handle. If the input sequence is longer
    than this, it will be truncated, defaults to 512 (optional)
    """
    encoded_question = tokenizer(
        question, max_length=max_length,
        padding="max_length",
        return_token_type_ids=True,
        truncation=True, return_tensors="pt")
    encoded_question.to(this_device)

    # reshaping (adding dimension) to have a batch size
    encoded_question.data["input_ids"] = torch.reshape(
        encoded_question.data["input_ids"],
        (1, encoded_question.data["input_ids"].shape[1]))
    encoded_question.data["token_type_ids"] = torch.reshape(
        encoded_question.data["token_type_ids"],
        (1, encoded_question.data[
            "token_type_ids"].shape[1]))
    encoded_question.data["attention_mask"] = torch.reshape(
        encoded_question.data["attention_mask"],
        (1, encoded_question.data[
            "attention_mask"].shape[1]))
    # calling the pre-trained language model UWB-AIR/MQDD-duplicates
    _, encoded_question_result = model(**encoded_question)

    return np.squeeze(encoded_question_result.detach().cpu().numpy().tolist())


def encode_questions(question, tokenizer, model, tokenized_question_example, this_device, batch_size=1, max_length=512):
    """
    This function encodes a given question using a tokenizer and a pre-trained language model, with a specified maximum
    length.

    :param this_device: cpu or cuda
    :param question: The input question that needs to be encoded
    :param tokenizer: The tokenizer is a tool used to convert text into numerical values that can be processed by model.
    :param model: The model parameter refers to the pre-trained language model.
    :param max_length: The maximum length of the input sequence that the model can handle. If the input sequence is longer
    than this, it will be truncated, defaults to 512 (optional)
    :param tokenized_question_example: It is an example of a tokenized question that will be used as a template to encode
    new questions. It is a list of integers representing the tokenized version of a question
    :param batch_size: The number of questions to be processed in a single batch. This can help speed up the encoding
    process by processing multiple questions at once, defaults to 1 (optional)
    """

    print("encoding...")
    encoded_result_list = []
    for i in tqdm(range(0, len(question), batch_size)):
        encoded_question = tokenizer(
            question[i:int(batch_size+i)], max_length=max_length,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True, return_tensors="pt")
        encoded_question.to(this_device)

        # reshaping (adding dimension) to have a batch size
        tokenized_question_example.data["input_ids"] = torch.reshape(
            encoded_question.data["input_ids"],
            (batch_size, encoded_question.data["input_ids"].shape[1]))
        tokenized_question_example.data["token_type_ids"] = torch.reshape(
            encoded_question.data["token_type_ids"],
            (batch_size, encoded_question.data[
                "token_type_ids"].shape[1]))
        tokenized_question_example.data["attention_mask"] = torch.reshape(
            encoded_question.data["attention_mask"],
            (batch_size, encoded_question.data[
                "attention_mask"].shape[1]))
        # calling the pre-trained language model UWB-AIR/MQDD-duplicates
        _, encoded_question_result = model(**tokenized_question_example)
        # unpacking results from batch size to list
        encoded_result_list.extend(encoded_question_result.detach().cpu().numpy().tolist())

    return np.array(encoded_result_list)

from django.apps import AppConfig

from faiss import read_index
from transformers import LongformerModel, AutoTokenizer


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

class SearchConfig(AppConfig):
    name = 'search'
    indexed_post_bodies = read_index("./search/indexed_data/Body.index")
    # todo replace with real key and access token
    INDEXED_SITE = "gamedev"
    STACK_EXCHANGE_KEY = 0
    STACK_EXCHANGE_ACCESS_TOKEN = 0
    tokenizer, model, tokenized_question_example, this_device = prepare_tok_model()



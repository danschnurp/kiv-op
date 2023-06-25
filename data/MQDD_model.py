import torch
from transformers import LongformerConfig, LongformerModel, AutoTokenizer

FIRST_DROPOUT = 0.4
SECOND_DROPOUT = 0.3
HIDDEN_DROPOUT = 0.3
ATTENTION_DROPOUT = 0.3


class ClsHeadModelMQDD(torch.nn.Module):

    def __init__(self, model_path):
        super(ClsHeadModelMQDD, self).__init__()
        # Transformers Longformer model
        self._model_config = LongformerConfig.from_pretrained(model_path,
                                                              hidden_dropout_prob=HIDDEN_DROPOUT,
                                                              attention_probs_dropout_prob=ATTENTION_DROPOUT)
        self._model_config.return_dict = False
        self.bert_model = LongformerModel.from_pretrained(model_path, config=self._model_config)
        self._dropout1 = torch.nn.Dropout(FIRST_DROPOUT)
        self._dropout2 = torch.nn.Dropout(SECOND_DROPOUT)
        self._dense = torch.nn.Linear(1536, 256, bias=True)
        self._out = torch.nn.Linear(256, 2, bias=False)
        self._relu = torch.nn.ReLU()
        self._softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        input_1, input_1_mask, input_1_tok_types, input_2, input_2_mask, input_2_tok_types = x.values()
        input_1, input_1_mask, input_1_tok_types = torch.squeeze(input_1), torch.squeeze(input_1_mask), torch.squeeze(
            input_1_tok_types)
        input_2, input_2_mask, input_2_tok_types = torch.squeeze(input_2), torch.squeeze(input_2_mask), torch.squeeze(
            input_2_tok_types)

        input_1, input_1_mask, input_1_tok_types = torch.reshape(input_1, (1, input_1.shape[0])), \
                                                   torch.reshape(input_1_mask, (1, input_1_mask.shape[0])), \
                                                   torch.reshape(input_1_tok_types, (1, input_1_tok_types.shape[0]))
        input_2, input_2_mask, input_2_tok_types = torch.reshape(input_2, (1, input_2.shape[0])), \
                                                   torch.reshape(input_2_mask, (1, input_2_mask.shape[0])), \
                                                   torch.reshape(input_2_tok_types, (1, input_2_tok_types.shape[0]))

        input_1_dense = self.bert_model(input_1, input_1_mask, None, token_type_ids=input_1_tok_types)[1]
        input_2_dense = self.bert_model(input_2, input_2_mask, None, token_type_ids=input_2_tok_types)[1]

        concat = torch.cat((input_1_dense, input_2_dense), 1)
        x = self._dropout1(concat)
        x = self._relu(self._dense(x))
        x = self._dropout2(x)
        x = self._out(x)
        x = x.detach().numpy()
        return "DUPLICATE" if x[0][0] < x[0][1] else "different"


def encode_classic(question, tokenizer):
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

    encoded_question.data["input_ids" + str(hash(question))] = torch.reshape(encoded_question.data["input_ids"],
                                                                             (1,
                                                                              encoded_question.data["input_ids"].shape[
                                                                                  0]))
    encoded_question.data["attention_mask" + str(hash(question))] = torch.reshape(
        encoded_question.data["attention_mask"],
        (1, encoded_question.data["attention_mask"].shape[0]))
    encoded_question.data["token_type_ids" + str(hash(question))] = torch.reshape(
        encoded_question.data["token_type_ids"],
        (1, encoded_question.data["token_type_ids"].shape[0]))

    del encoded_question.data["input_ids"]
    del encoded_question.data["attention_mask"]
    del encoded_question.data["token_type_ids"]

    return encoded_question


def load_nett(model_location: str) -> tuple:
    """
    > Loads a model from a given location and returns the model and the model's input shape

    :param model_location: The location of the model file
    :type model_location: str
    """
    # Loading the pretrained model from HuggingFace.
    tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")

    # tokenizer eos_token -> sep , bos_token -> cls
    tokenizer.bos_token = "cls"
    tokenizer.eos_token = "sep"

    model = ClsHeadModelMQDD("UWB-AIR/MQDD-duplicates")

    # https://drive.google.com/drive/folders/1CYiqF2GJ2fSQzx_oM4-X_IhpObi4af5Q
    # Loading the model from the location specified in `model_location` and mapping it to the CPU.
    ckpt = torch.load(model_location, map_location="cpu")
    # A workaround for a bug in the HuggingFace library.
    position_ids = ckpt["model_state"]["bert_model.embeddings.position_ids"]
    del ckpt["model_state"]["bert_model.embeddings.position_ids"]
    model.load_state_dict(ckpt["model_state"])
    # Returning the model, the tokenizer, and the position ids.
    return model, tokenizer, position_ids

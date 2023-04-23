import torch


def encode_question(question, tokenizer):
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

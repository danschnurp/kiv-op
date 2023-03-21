import torch
from transformers import LongformerConfig, LongformerModel

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

    def forward(self, input_1, input_2):
        input_1_dense = self.bert_model(torch.squeeze(input_1, 1), None)
        input_2_dense = self.bert_model(torch.squeeze(input_2, 1), None)

        concat = torch.cat((input_1_dense[1], input_2_dense[1]), 1)
        x = self._dropout1(concat)
        x = self._relu(self._dense(x))
        x = self._dropout2(x)
        x = self._out(x)
        x = x.detach().numpy()
        return "DUPLICATE" if x[0][0] < x[0][1] else "different"

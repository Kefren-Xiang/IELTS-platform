# bert_score_model.py
import torch.nn as nn
from transformers import AutoModel

class BertRegressor(nn.Module):
    def __init__(self, model_name, n_outputs=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.regressor = nn.Linear(hidden, n_outputs)
        nn.init.xavier_uniform_(self.regressor.weight)
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.pooler_output  # shape: [batch, hidden]
        preds = self.regressor(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(preds, labels)
        return {"loss": loss, "logits": preds}

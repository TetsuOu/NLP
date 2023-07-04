import torch
from torch import nn
from transformers import AutoModel
from transformers import BertPreTrainedModel, BertModel


checkpoint = "bert-base-chinese"

# 这种方式简单粗暴，但是相当于在 Transformers 模型外又包了一层，因此无法再调用 Transformers 库预置的模型函数
class BertForPairwiseCLS(nn.Module):
    def __init__(self):
        super(BertForPairwiseCLS, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits

# 更为常见的写法是继承 Transformers 库中的预训练模型来创建自己的模型。
# 例如这里可以继承 BERT 模型（BertPreTrainedModel 类）来创建一个与上面模型结构完全相同的分类器：
class BertForPairwiseCLS_V2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()
    
    def forward(self, x):
        bert_output = self.bert(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model = BertForPairwiseCLS().to(device)
    # model_2 = BertForPairwiseCLS_V2().to(device)
    print(model)
import torch
from model import BertForPairwiseCLS
from transformers import AutoTokenizer

model_path = 'epoch_3_valid_acc_73.0_model_weights.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = BertForPairwiseCLS()
model.load_state_dict(torch.load(model_path))
model = model.to(device)

def predict(sent_1, sent_2, model, tokenizer):
    
    inputs = tokenizer(
        sent_1, 
        sent_2, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs
    pred = int(logits.argmax(dim=-1)[0].cpu().numpy())
    prob = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
    return pred, prob[pred]



if __name__ == '__main__':
    # print(predict('还款还清了，为什么花呗账单显示还要还款', '花呗全额还清怎么显示没有还款',model, tokenizer))
    import json
    
    size = 0
    correct = 0
    
    with open('语义相似度/raw_data/dev.json', 'rt') as f:
        for idx, line in enumerate(f):
            sample = json.loads(line.strip())
            label = int(sample['label'])
            pred, prob = predict(sample['sentence1'], sample['sentence2'], model, tokenizer)
            size += 1
            if(pred == label):
                correct += 1
    
    print(f'acc: {correct/size}')
            
            
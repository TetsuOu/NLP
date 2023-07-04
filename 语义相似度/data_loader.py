from torch.utils.data import Dataset,IterableDataset
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#如果数据集非常巨大，难以一次性加载到内存中
#可以继承 IterableDataset 类构建迭代型数据集

class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, 'rt') as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, 
        batch_sentence_2, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y

def AFQMC_Dataloader(path, shuffle=True, batch_size=32, num_work=2):
    data = AFQMC(path)
    return DataLoader(dataset=data,
                      shuffle=shuffle,
                      batch_size=batch_size,
                      num_workers=num_work,
                      collate_fn=collote_fn)
    
  
if __name__ == '__main__':
    # train_data = AFQMC('语义相似度/raw_data/train.json')
    # print(train_data[0])
    
    # train_data = IterableAFQMC('语义相似度/raw_data/train.json')
    # print(next(iter(train_data)))

    # train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
    
    train_dataloader = AFQMC_Dataloader(path='语义相似度/raw_data/train.json',batch_size=4, shuffle=False)
    
    batch_X, batch_y = next(iter(train_dataloader))
    print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
    print('batch_y shape:', batch_y.shape)
    print(batch_X)
    print(batch_y)
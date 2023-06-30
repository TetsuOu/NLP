import torch
from torch.utils.data import Dataset, DataLoader

class Mydataset(Dataset):
    def __init__(self, root_dir):
        self.data, self.label = torch.load(root_dir)

    def __getitem__(self, item):
        ##[batch_size, len, dim]
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]

def collate(samples):
    sent_list = []
    label_list = []
    n = len(samples)
    for i in range(n):
        sent, label = samples[i]
        sent_list.append(sent)
        label_list.append(label)
    return torch.stack(sent_list, dim=0), torch.stack(label_list, dim=0)

def Mydataloader(path, shuffle=True, batch_size=32, num_works=2):
    data = Mydataset(path)
    return DataLoader(dataset=data,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_works,
                      collate_fn=collate)

if __name__ == '__main__':
    data_loader_train = Mydataloader(path='data/train/train.buffer', shuffle=True, batch_size=32, num_works=2)
    for batch_idx, x in enumerate(data_loader_train):
        data, label = x
        break
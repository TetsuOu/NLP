
## 语义相似度

使用transforms库中的Bert

### dataset

AFQMC: 蚂蚁金融语义相似度数据集
[https://tianchi.aliyun.com/dataset/106411?t=1688452061335](https://tianchi.aliyun.com/dataset/106411?t=1688452061335)

### data_loader

DataLoader 按照设置的 batch size 每次对 4 个样本进行编码，并且通过设置 padding=True 和 truncation=True 来自动对每个 batch 中的样本进行补全和截断。这里选择 BERT 模型作为 checkpoint，所以每个样本都被处理成了“[CLS] sne1 [SEP] sen2 [SEP]”的形式。

```python
train_dataloader = AFQMC_Dataloader(path='语义相似度/raw_data/train.json',batch_size=4, shuffle=True)
    
batch_X, batch_y = next(iter(train_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print('batch_y shape:', batch_y.shape)
print(batch_X)
print(batch_y)
```

```bash
batch_X shape: {'input_ids': torch.Size([4, 30]), 'token_type_ids': torch.Size([4, 30]), 'attention_mask': torch.Size([4, 30])}
batch_y shape: torch.Size([4])
{'input_ids': tensor([[ 101, 6010, 6009,  955, 1446, 5023, 7583, 6820, 3621, 1377,  809, 2940,
         2768, 1044, 2622, 1400, 3315, 1408,  102,  955, 1446, 3300, 1044, 2622,
         1168, 3309, 6820, 3315, 1408,  102],
        [ 101, 6010, 6009, 5709, 1446, 6432, 2769, 6824, 5276,  671, 3613,  102,
         6010, 6009, 5709, 1446, 6824, 5276, 6121,  711, 3221,  784,  720,  102,
            0,    0,    0,    0,    0,    0],
        [ 101, 2376, 2769, 4692,  671,  678, 3315, 3299, 5709, 1446, 6572, 1296,
         3300, 3766, 3300, 5310, 3926,  102,  678, 3299, 5709, 1446, 6572, 1296,
          102,    0,    0,    0,    0,    0],
        [ 101, 6010, 6009,  955, 1446, 1914, 7270, 3198, 7313, 5341, 1394, 6397,
          844,  671, 3613,  102,  955, 1446, 2533, 6397,  844, 1914,  719,  102,
            0,    0,    0,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
         1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0]])}
tensor([0, 0, 0, 0])

```

101 -> [CLS]
102 -> [SEP]

### 参考链接

[https://transformers.run/intro/2021-12-17-transformers-note-4](https://transformers.run/intro/2021-12-17-transformers-note-4)
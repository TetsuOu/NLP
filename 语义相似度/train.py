import torch
import torch.nn as nn
from model import BertForPairwiseCLS
from transformers import AdamW, get_scheduler
from data_loader import AFQMC_Dataloader
from loop import train_loop, test_loop

learning_rate = 1e-5
epoch_num = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
model = BertForPairwiseCLS()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)

train_dataloader = AFQMC_Dataloader(path='语义相似度/raw_data/train.json',batch_size=4, shuffle=False)
valid_dataloader = AFQMC_Dataloader(path='语义相似度/raw_data/dev.json',batch_size=4, shuffle=False)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_acc = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    valid_acc = test_loop(valid_dataloader, model, mode='Valid')
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
print("Done!")
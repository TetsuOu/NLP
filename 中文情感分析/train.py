import torch
from data_loader import Mydataloader
from model import CNN,lstm

batch_size = 32
n_filters = 100
filter_sizes = [2,3,4]
dropout = 0.5
n_epoch = 10
vocab_dim = 100
lr = 0.001

if __name__ == '__main__':
    data_loader_train = Mydataloader(path='data/train/train.buffer', shuffle=True, batch_size=32, num_works=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (f'正在使用计算的是：{device}')

    # model = CNN(vocab_dim, n_filters, filter_sizes, dropout)
    model = lstm(embedding_dim= vocab_dim, hidden_dim=128)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss_func = torch.nn.CrossEntropyLoss()
    from sklearn.metrics import accuracy_score, classification_report
    print('Start train!')
    best_acc = 0

    model.train()
    for epoch in range(n_epoch):
        total = 0
        correct = 0
        epoch_loss = 0
        for batch_idx, x in enumerate(data_loader_train):
            data, label = x
            data = torch.as_tensor(data, dtype=torch.float32)
            target = label.long()

            data, target = data.to(device), target.to(device)
            # output = model(data)
            output, h_state = model(data)
            correct += int(torch.sum(torch.argmax(output, dim=1) == target))
            total += len(target)
            optimizer.zero_grad()
            loss = loss_func(output, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        acc = correct*100/total
        loss = epoch_loss/(batch_idx+1)
        if(acc>best_acc):
            best_acc = acc
            # torch.save(model, f'cnn.pt')
            torch.save(model, f'lstm.pt')
        if(epoch%1==0):
            print(f'Epoch {epoch} accuracy: {acc} best acc: {best_acc} loss: {loss}')


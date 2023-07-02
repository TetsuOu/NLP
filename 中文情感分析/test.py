import torch
from data_loader import Mydataloader
from model import CNN

if __name__ == '__main__':
    # model = torch.load('cnn.pt')
    model = torch.load('lstm.pt')
    data_loader_test = Mydataloader(path='data/test/test.buffer', shuffle=True, batch_size=32, num_works=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'正在使用计算的是：{device}')
    for epoch in range(1):

        model.eval()
        total = 0
        correct = 0
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader_test):
            data = torch.as_tensor(data, dtype=torch.float32)
            target = target.long()

            data, target = data.to(device), target.to(device)
            # output = model(data)
            output, h_state = model(data)
            correct += int(torch.sum(torch.argmax(output, dim=1) == target))
            total += len(target)

        acc = correct * 100 / total
        loss = epoch_loss / (batch_idx + 1)

        print(f'Test   accuracy: {acc} loss: {loss}')
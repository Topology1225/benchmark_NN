import torch
import torchvision


def main():
    
    # 学習パラメータ設定
    EPOCH = 10

    # dataloaderの定義
    dataset = \
        torchvision.datasets.CIFAR10(
            root="./", 
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True, num_workers=0,
        pin_memory=True
    )

    # modelの設定
    model = torch.hub.load(
        repo_or_dir="pytorch/vision",
        model="resnet50",
        pretrained=False
    )

    # optimizerの設定
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-3, momentum=0.9
    )

    # 損失関数の設定
    criterion = torch.nn.CrossEntropyLoss()

    # cuda
    device = torch.device("cuda:0")
    model = model.to(device)

    # 学習開始
    for epoch in range(EPOCH):

        acc  = 0.
        lt_loss = list()

        for x, y in dataloader:
            with torch.cuda.amp.autocast():
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss   = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc += ((logits.argmax(dim=1) == y).sum() / x.shape[0]).item()
            lt_loss.append(loss.mean().cpu().item())
        
        msg = (
            f"epoch{epoch} ",
            f"loss: {torch.mean(torch.tensor(lt_loss)):.4f}",
            f"acc : {acc:.4f}"
        )
        print("   ".join(msg))




if __name__=="__main__":
    main()
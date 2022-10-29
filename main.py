from mlp_mixer import MLPMixer
import torch
import torchvision
import numpy as np
from tqdm import tqdm


def main():

    '''
    mlp = MLP(50, 30)

    inputs = torch.rand(16, 40, 50)
    print(inputs.shape)
    out = mlp(inputs)
    print(out.shape)


    mixer = Mixer(50, 30)
    out = mixer(inputs)
    print(inputs.shape, out.shape)
    '''

    # Initialize model
    model = MLPMixer(img_shape=(1, 28, 28), patch_size=7, embedding_dim=128,
                     n_layers=3, mlp_dim=64, clf_dim=512, n_classes=10)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()

    model.train()

    # Training parameters
    epochs = 300
    batch_size = 64


    # Load data
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomRotation(15)
    ])

    test_transform = torchvision.transforms.ToTensor()

    root = 'C:/Users/Dan/data'
    train_dataset = torchvision.datasets.MNIST(root=root, download=True, train=True, transform=train_transform)
    test_dataset = torchvision.datasets.MNIST(root=root, download=True, train=False, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):

        train_loss, test_loss, train_acc, test_acc = [], [], [], []

        # Train epoch
        model.train()
        for i, (x, labels) in enumerate(tqdm(trainloader)):

            optimizer.zero_grad()

            out = model(x)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())

            # Compute accuracy
            preds = torch.argmax(out, dim=1)
            acc = torch.sum(labels == preds).detach().numpy() / len(x)
            train_acc.append(acc)

        # Eval on test set
        model.eval()
        for i, (x, labels) in enumerate(tqdm(testloader)):

            with torch.no_grad():
                out = model(x)
                loss = loss_func(out, labels)

            test_loss.append(loss.detach().numpy())

            preds = torch.argmax(out, dim=1)
            acc = torch.sum(labels == preds).detach().numpy() / len(x)
            test_acc.append(acc)

        train_loss = np.mean(train_loss)
        test_loss = np.mean(test_loss)
        train_acc = np.mean(train_acc)
        test_acc = np.mean(test_acc)

        print('Epoch: {}  Train loss: {}  Test loss: {}  Train acc: {}  Test acc: {}'
              .format(epoch, train_loss, test_loss, train_acc, test_acc))

if __name__ == '__main__':
    main()
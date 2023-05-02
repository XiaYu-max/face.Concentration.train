from FaceCNN import *
import torch.utils.data as data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#classes = ('Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral')
classes = ('focus', 'unfocus')
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    
    # 
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(7)]
    n_class_samples = [0 for i in range(7)]

    for images, labels in val_loader:
        # gpu
        images = images.to(device)
        labels = labels.to(device)

        outputs = model.forward(images)
        # pred = np.argmax(pred.data.cpu().numpy(), axis = 1)

        # gpu
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

        num += len(images)
        for i in range(len(images)):
            if predicted[i] == labels[i]:
                n_class_correct[labels[i]] += 1
            n_class_samples[labels[i]] += 1 

    acc = 100.0 * n_correct / num
    print(f'Accuracy of the network: {acc} %')
    
    for i in range(0, len(classes)):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'{classes[i]} 有 n_class_samples = {n_class_samples[i]}')
        print(f'{classes[i]} 有 n_class_correct = {n_class_correct[i]}')
        print(f'Accuracy of {i}: {acc} %')
    return acc


def train(train_dataset, batch_size, epochs, learning_rate, wt_decay):

    train_loader = data.DataLoader(train_dataset, batch_size)

    
    model = FaceCNN().to(device)
    # model = FaceCNN()

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr = learning_rate, weight_decay = wt_decay)

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.8)
    
    for epoch in range(epochs):
        num  = 0
        loss_rate = 0

        #scheduler.step()

        model.train()
        for images, labels in train_loader:

          #  gpu
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            output = model(images)

            loss_rate = loss_function(output, labels )  

            loss_rate.backward()

        #    utils.clip_gradient(optimizer, 0.1)
            
            optimizer.step()
        print('After{} epochs , the loss_rate is : '.format(epoch + 1), loss_rate.item())

        if loss_rate.item() == 0.0 :
            break
        # if(loss_rate.item() - num > 0.05):
        #     break
        # else:
        #     num = loss_rate.item()


    return model
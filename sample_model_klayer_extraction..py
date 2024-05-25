import torch
import torch.nn as nn


def getRetrainLayers(m, name, ret):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
        ret.append((m, name))
        #print(name)
    for child_name, child in m.named_children():
        getRetrainLayers(child, f'{name}.{child_name}', ret)
    return ret


def resetFinalResnet(model, num_retrain):
    for param in model.parameters():
        param.requires_grad = False
    done = 0
    ret = getRetrainLayers(model, 'M', [])
    ret.reverse()
    for idx in range(len(ret)):
        if isinstance(ret[idx][0], nn.Conv2d) or isinstance(ret[idx][0], nn.Linear):
            done += 1
        for param in ret[idx][0].parameters():
            param.requires_grad = True
        if done >= num_retrain:
            break

    return model



# Assuming get_network and other necessary imports are defined elsewhere
model = get_network(arg)
model = torch.nn.DataParallel(model)
checkpoint = torch.load(arg.checkpoint_load)
model.load_state_dict(checkpoint['model'])
k_layer = 2
model = resetFinalResnet(model, k_layer)
# do args to device if we have cudd

train(model, epoch, )

def trainable_params_(m):
    return [p for p in m.parameters() if p.requires_grad]


def train(model, epochs, train_loader):

    optimizer = torch.optim.SGD(trainable_params_(model), lr=arg.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg.schedule, gamma=arg.gamma)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for data, targets in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


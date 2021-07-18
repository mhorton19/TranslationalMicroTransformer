
import torch
from torch.nn.parameter import Parameter
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
import random
import numpy as np
from torch.nn import init

VECTOR_SIZE = 9
NCLASSES = 10


def validation(model, testloader, criterion):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        accuracy = 0
        iterations = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model.forward(images)
            test_loss += criterion(outputs, labels).item()

            outputs_np = np.argmax(outputs.data.cpu().numpy(), 1)
            #print(outputs_np)
            correct = np.mean(outputs_np == labels.data.cpu().numpy())
            accuracy += correct
            iterations += 1

        model.train()
        return test_loss/iterations, accuracy/iterations

if __name__ == '__main__':
    device = torch.device("cuda:0")
    bs = 16
    #torch.multiprocessing.freeze_support()
    aug_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.RandomCrop(size=[32,32], padding=4),
         transforms.RandomAffine(180, scale=(0.8, 1.2), shear=(20, 20)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='cifar-10-python', train=True,
                                            download=True, transform=aug_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='cifar-10-python', train=False,
                                           download=True, transform=aug_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                             shuffle=False, num_workers=0)


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()


    import torch.nn as nn
    import torch.nn.functional as F


    class DilatedCNN(nn.Module):
        def __init__(self):
            super(DilatedCNN, self).__init__()

            num_kernels = 64
            # groups = num_kernels//16
            groups = 1
            self.num_kernels = num_kernels
            self.conv1_1 = nn.Conv2d(3, num_kernels, 3, padding=1, bias=False)
            self.bn1_1 = nn.BatchNorm2d(num_kernels)
            self.cond_conv1_1 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn1_1 = nn.BatchNorm2d(num_kernels)
            self.conv1_2 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, bias=False)
            self.bn1_2 = nn.BatchNorm2d(num_kernels)
            self.avg_pool_1 = nn.AvgPool2d(2)

            self.cond_conv2_1 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn2_1 = nn.BatchNorm2d(num_kernels)
            self.conv2_1 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1)
            self.bn2_1 = nn.BatchNorm2d(num_kernels)
            self.avg_pool_2 = nn.AvgPool2d(2)

            self.cond_conv3_1 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn3_1 = nn.BatchNorm2d(num_kernels)

            self.cond_conv3_2 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn3_2 = nn.BatchNorm2d(num_kernels)

            self.cond_conv3_3 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn3_3 = nn.BatchNorm2d(num_kernels)

            self.global_pool = nn.AvgPool2d(8, 8)
            self.fc = nn.Linear(num_kernels, NCLASSES)

        def forward(self, x):
            x = F.relu(self.bn1_1(self.conv1_1(x)))
            x = F.relu(self.cond_bn1_1(self.cond_conv1_1(x)))
            x = F.relu(self.avg_pool_1(self.bn1_2(self.conv1_2(x))))

            x = F.relu(self.cond_bn2_1(self.cond_conv2_1(x)))
            x = F.relu(self.avg_pool_2(self.bn2_1(self.conv2_1(x))))

            x = F.relu(self.cond_bn3_1(self.cond_conv3_1(x)))
            x = F.relu(self.cond_bn3_2(self.cond_conv3_2(x)))
            x = F.relu(self.cond_bn3_3(self.cond_conv3_3(x)))
            x = self.global_pool(x).view(-1, self.num_kernels)
            x = self.fc(x)
            return x

    class BaselineCNN(nn.Module):
        def __init__(self):
            super(BaselineCNN, self).__init__()
            
            num_kernels = 64
            #groups = num_kernels//16
            groups = 1
            self.num_kernels = num_kernels
            self.conv1_1 = nn.Conv2d(3, num_kernels, 3, padding=1, bias=False)
            self.bn1_1 = nn.BatchNorm2d(num_kernels)
            self.cond_conv1_1 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn1_1 = nn.BatchNorm2d(num_kernels)
            self.conv1_2 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, bias=False)
            self.bn1_2 = nn.BatchNorm2d(num_kernels)
            self.avg_pool_1 = nn.AvgPool2d(2)
            
            self.cond_conv2_1 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn2_1 = nn.BatchNorm2d(num_kernels)
            self.conv2_1 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1)
            self.bn2_1 = nn.BatchNorm2d(num_kernels)
            self.avg_pool_2 = nn.AvgPool2d(2)

            self.cond_conv3_1 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn3_1 = nn.BatchNorm2d(num_kernels)
            
            self.cond_conv3_2 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn3_2 = nn.BatchNorm2d(num_kernels)
            
            self.cond_conv3_3 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1, groups=groups, bias=False)
            self.cond_bn3_3 = nn.BatchNorm2d(num_kernels)
            
            self.global_pool = nn.AvgPool2d(8, 8)
            self.fc = nn.Linear(num_kernels, NCLASSES)
            
        def forward(self, x):   
            x = F.relu(self.bn1_1(self.conv1_1(x)))
            x = F.relu(self.cond_bn1_1(self.cond_conv1_1(x)))
            x = F.relu(self.avg_pool_1(self.bn1_2(self.conv1_2(x))))

            x = F.relu(self.cond_bn2_1(self.cond_conv2_1(x)))
            x = F.relu(self.avg_pool_2(self.bn2_1(self.conv2_1(x))))
            
            x = F.relu(self.cond_bn3_1(self.cond_conv3_1(x)))
            x = F.relu(self.cond_bn3_2(self.cond_conv3_2(x)))
            x = F.relu(self.cond_bn3_3(self.cond_conv3_3(x)))
            x = self.global_pool(x).view(-1, self.num_kernels)
            x = self.fc(x)
            return x

    class TranslationalConv2d(nn.Module):
        def __init__(self, in_size, out_size, window_size, num_heads, attention_len, value_len, dilation=1):
            super(TranslationalConv2d, self).__init__()
            self.dilation = dilation
            self.num_heads = num_heads
            self.value_len = value_len
            self.attention_len = attention_len

            self.attention_vec_len = num_heads * attention_len
            self.attention_out_vec_len = num_heads * value_len

            self.window_size = window_size

            self.key_positional_bias = torch.nn.Parameter(torch.empty(1, 1, 1, self.num_heads, self.window_size ** 2, self.attention_len))

            self.key_gen = torch.nn.Linear(in_size, self.attention_vec_len, bias=False)
            self.query_gen = torch.nn.Linear(in_size, self.attention_vec_len)
            self.value_gen = torch.nn.Linear(in_size, self.attention_out_vec_len)

            self.output_gen = torch.nn.Linear(self.attention_out_vec_len, out_size)

            self.reset_parameters()

        def reset_parameters(self) -> None:
            init.normal_(self.key_positional_bias)

        def forward(self, x, residual=0):
            surfaces = []
            starting_point = -1 * (self.window_size // 2) * self.dilation
            for x_idx in range(self.window_size):
                for y_idx in range(self.window_size):
                    x_offset = starting_point + (x_idx * self.dilation)
                    y_offset = starting_point + (y_idx * self.dilation)

                    if y_offset > 0:
                        surface = F.pad(x,  [0, y_offset])[:,:,:,-x.shape[-1]:]
                    elif y_offset < 0:
                        surface = F.pad(x, [-1*y_offset, 0])[:,:,:,:x.shape[-1]]
                    else:
                        surface = x

                    if x_offset > 0:
                        surface = F.pad(surface,  [0, 0] + [0, x_offset])[:,:,-x.shape[-2]:,:]
                    elif x_offset < 0:
                        surface = F.pad(surface,  [0, 0] + [-1*x_offset, 0])[:,:,:x.shape[-2],:]
                    else:
                        surface = surface

                    surface = surface.unsqueeze(-1)

                    surfaces.append(surface)

            stacked_pixels = torch.cat(surfaces, axis=-1)
            stacked_pixels = stacked_pixels.permute(0, 2, 3, 4, 1)
            values = self.value_gen(stacked_pixels)
            values = values.view(values.shape[:-1] + (self.num_heads, self.value_len))
            values = values.permute(0, 1, 2, 4, 3, 5)

            keys = self.key_gen(stacked_pixels)
            keys = keys.view(keys.shape[:-1] + (self.num_heads, self.attention_len))
            keys = keys.permute(0, 1, 2, 4, 3, 5)
            keys = (keys + self.key_positional_bias) / math.sqrt(2)

            non_stacked_pixels = x.permute(0, 2, 3, 1)
            queries = self.query_gen(non_stacked_pixels)
            queries = queries.view(queries.shape[:-1] + (self.num_heads, self.attention_len, 1))

            attention_outputs = torch.matmul(keys, queries).squeeze(-1) / math.sqrt(self.attention_len)

            attention_outputs = F.softmax(attention_outputs, dim=-2).unsqueeze(-2)

            agg_values = torch.matmul(attention_outputs, values)
            agg_values = agg_values.view(*agg_values.shape[:3], -1)

            out_vals = self.output_gen(agg_values)
            out_vals = out_vals.permute(0, 3, 1, 2)

            return out_vals + residual





    class TranslationalConvCNN(nn.Module):
        def __init__(self):
            super(TranslationalConvCNN, self).__init__()

            num_kernels = 64
            # groups = num_kernels//16
            groups = 1
            self.num_kernels = num_kernels
            self.conv1_1 = TranslationalConv2d(3, num_kernels, 3, 10, 10, 15)
            self.bn1_1 = nn.BatchNorm2d(num_kernels)
            self.conv1_2 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 15, dilation=2)
            self.bn1_2 = nn.BatchNorm2d(num_kernels)
            self.conv1_3 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 15, dilation=4)
            self.bn1_3 = nn.BatchNorm2d(num_kernels)
            self.conv1_4 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 15, dilation=8)
            self.bn1_4 = nn.BatchNorm2d(num_kernels)
            self.conv1_5 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 15, dilation=16)
            self.bn1_5 = nn.BatchNorm2d(num_kernels)

            self.conv2_1 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 15)
            self.bn2_1 = nn.BatchNorm2d(num_kernels)
            self.conv2_2 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 15, dilation=2)
            self.bn2_2 = nn.BatchNorm2d(num_kernels)
            self.conv2_3 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 15, dilation=4)
            self.bn2_3 = nn.BatchNorm2d(num_kernels)
            self.conv2_4 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 15, dilation=8)
            self.bn2_4 = nn.BatchNorm2d(num_kernels)
            self.conv2_5 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 15, dilation=16)
            self.bn2_5 = nn.BatchNorm2d(num_kernels)


            self.global_pool = nn.AvgPool2d(32, 32)
            self.fc = nn.Linear(num_kernels, NCLASSES)

        def forward(self, x):
            res = self.conv1_1(x)
            x = F.relu(self.bn1_1(res))
            res = self.conv1_2(x, residual=res)
            x = F.relu(self.bn1_2(res))
            res = self.conv1_3(x, residual=res)
            x = F.relu(self.bn1_3(res))
            res = self.conv1_4(x, residual=res)
            x = F.relu(self.bn1_4(res))
            res = self.conv1_5(x, residual=res)
            x = F.relu(self.bn1_5(res))

            res = self.conv2_1(x, residual=res)
            x = F.relu(self.bn2_1(res))
            res = self.conv2_2(x, residual=res)
            x = F.relu(self.bn2_2(res))
            res = self.conv2_3(x, residual=res)
            x = F.relu(self.bn2_3(res))
            res = self.conv2_4(x, residual=res)
            x = F.relu(self.bn2_4(res))
            res = self.conv2_5(x, residual=res)
            x = F.relu(self.bn2_5(res))

            x = self.global_pool(x).view(-1, self.num_kernels)
            x = self.fc(x)
            return x

    class BaselineDilatedCNN(nn.Module):
        def __init__(self):
            super(BaselineDilatedCNN, self).__init__()

            num_kernels = 64
            # groups = num_kernels//16
            groups = 1
            self.num_kernels = num_kernels
            self.conv1_1 = nn.Conv2d(3, num_kernels, 3, padding=1)
            self.bn1_1 = nn.BatchNorm2d(num_kernels)
            self.conv1_2 = nn.Conv2d(num_kernels, num_kernels, 3, dilation=2, padding=2)
            self.bn1_2 = nn.BatchNorm2d(num_kernels)
            self.conv1_3 = nn.Conv2d(num_kernels, num_kernels, 3, dilation=4, padding=4)
            self.bn1_3 = nn.BatchNorm2d(num_kernels)
            self.conv1_4 = nn.Conv2d(num_kernels, num_kernels, 3, dilation=8, padding=8)
            self.bn1_4 = nn.BatchNorm2d(num_kernels)
            self.conv1_5 = nn.Conv2d(num_kernels, num_kernels, 3, dilation=16, padding=16)
            self.bn1_5 = nn.BatchNorm2d(num_kernels)

            self.conv2_1 = nn.Conv2d(num_kernels, num_kernels, 3, padding=1)
            self.bn2_1 = nn.BatchNorm2d(num_kernels)
            self.conv2_2 = nn.Conv2d(num_kernels, num_kernels, 3, dilation=2, padding=2)
            self.bn2_2 = nn.BatchNorm2d(num_kernels)
            self.conv2_3 = nn.Conv2d(num_kernels, num_kernels, 3, dilation=4, padding=4)
            self.bn2_3 = nn.BatchNorm2d(num_kernels)
            self.conv2_4 = nn.Conv2d(num_kernels, num_kernels, 3, dilation=8, padding=8)
            self.bn2_4 = nn.BatchNorm2d(num_kernels)
            self.conv2_5 = nn.Conv2d(num_kernels, num_kernels, 3, dilation=16, padding=16)
            self.bn2_5 = nn.BatchNorm2d(num_kernels)

            self.global_pool = nn.AvgPool2d(32, 32)
            self.fc = nn.Linear(num_kernels, NCLASSES)

        def forward(self, x):
            res = self.conv1_1(x)
            x = F.relu(self.bn1_1(res))
            res = res + self.conv1_2(x)
            x = F.relu(self.bn1_2(res))
            res = res + self.conv1_3(x)
            x = F.relu(self.bn1_3(res))
            res = res + self.conv1_4(x)
            x = F.relu(self.bn1_4(res))
            res = res + self.conv1_5(x)
            x = F.relu(self.bn1_5(res))

            res = res + self.conv2_1(x)
            x = F.relu(self.bn2_1(res))
            res = res + self.conv2_2(x)
            x = F.relu(self.bn2_2(res))
            res = res + self.conv2_3(x)
            x = F.relu(self.bn2_3(res))
            res = res + self.conv2_4(x)
            x = F.relu(self.bn2_4(res))
            res = res + self.conv2_5(x)
            x = F.relu(self.bn2_5(res))

            x = self.global_pool(x).view(-1, self.num_kernels)
            x = self.fc(x)
            return x

    class TranslationalConvCNN2(nn.Module):
        def __init__(self):
            super(TranslationalConvCNN2, self).__init__()

            num_kernels = 64
            # groups = num_kernels//16
            groups = 1
            self.num_kernels = num_kernels
            self.conv1_1 = TranslationalConv2d(3, num_kernels, 3, 11, 10, 10)
            self.bn1_1 = nn.BatchNorm2d(num_kernels)
            self.cond_conv1_1 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 10)
            self.cond_bn1_1 = nn.BatchNorm2d(num_kernels)
            self.conv1_2 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 10)
            self.bn1_2 = nn.BatchNorm2d(num_kernels)
            self.avg_pool_1 = nn.AvgPool2d(2)

            self.cond_conv2_1 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 10)
            self.cond_bn2_1 = nn.BatchNorm2d(num_kernels)
            self.conv2_1 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 10)
            self.bn2_1 = nn.BatchNorm2d(num_kernels)
            self.avg_pool_2 = nn.AvgPool2d(2)

            self.cond_conv3_1 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 10)
            self.cond_bn3_1 = nn.BatchNorm2d(num_kernels)

            self.cond_conv3_2 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 10)
            self.cond_bn3_2 = nn.BatchNorm2d(num_kernels)

            self.cond_conv3_3 = TranslationalConv2d(num_kernels, num_kernels, 3, 10, 10, 10)
            self.cond_bn3_3 = nn.BatchNorm2d(num_kernels)

            self.global_pool = nn.AvgPool2d(8, 8)
            self.fc = nn.Linear(num_kernels, NCLASSES)

        def forward(self, x):
            res = self.conv1_1(x)
            x = F.relu(self.bn1_1(res))
            res = self.cond_conv1_1(x, residual=res)
            x = F.relu(self.cond_bn1_1(res))
            res = self.avg_pool_1(self.conv1_2(x, residual=res))
            x = F.relu(self.bn1_2(res))

            res = self.cond_conv2_1(x, residual=res)
            x = F.relu(self.cond_bn2_1(res))
            res = self.avg_pool_2(self.conv2_1(x, residual=res))
            x = F.relu(self.bn2_1(res))

            res = self.cond_conv3_1(x, residual=res)
            x = F.relu(self.cond_bn3_1(res))
            res = self.cond_conv3_2(x, residual=res)
            x = F.relu(self.cond_bn3_2(res))
            res = self.cond_conv3_3(x, residual=res)
            x = F.relu(self.cond_bn3_3(res))
            x = self.global_pool(x).view(-1, self.num_kernels)
            x = self.fc(x)
            return x


    net = TranslationalConvCNN()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    net = net.to(device)

    
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), amsgrad=True)

    print_increment = 100
    for epoch in range(300):  # loop over the dataset multiple times
        running_loss = 0.0
        running_accuracy = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            outputs_np = np.argmax(outputs.data.cpu().numpy(), 1)

            correct = np.mean(outputs_np == labels.data.cpu().numpy())
            running_accuracy += correct
            if i % print_increment == print_increment-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f accuracy %.5f' %
                      (epoch + 1, i + 1, running_loss / print_increment, running_accuracy/print_increment))
                running_loss = 0.0
                running_accuracy = 0.0

             

        testdataiter = iter(testloader)
        (loss, acc) = validation(net, testdataiter, criterion)
        
        print('EPOCH %d VAL loss: %.5f accuracy %.5f' %
                      (epoch + 1, loss, acc))



        

    print('Finished Training')

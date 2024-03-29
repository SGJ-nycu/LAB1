import os
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn import metrics


def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

def plot_accuracy(train_acc_list, train_acc_list2, train_acc_list3):
    plt.plot(train_acc_list)
    plt.plot(train_acc_list2)
    plt.plot(train_acc_list3)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['model1', 'model2', 'model3'], loc='best')
    plt.show()

def plot_f1_score(f1_score_list, f1_score_list2, f1_score_list3):
    plt.plot(f1_score_list_train)
    plt.plot(f1_score_list_train2)
    plt.plot(f1_score_list_train3)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['model1', 'model2', 'model3'], loc='best')
    plt.show()

def plot_confusion_matrix_model1(best_c_matrix):
    TP = best_c_matrix[0][0]
    FP = best_c_matrix[0][1]
    TN = best_c_matrix[1][0]
    FN = best_c_matrix[1][1]
    print("Confusion Matrix for model 1 : ")
    print(f"[{TP}] [{FP}]")
    print(f"[{FN}] [{TN}]")

def plot_confusion_matrix_model2(best_c_matrix2):
    TP = best_c_matrix2[0][0]
    FP = best_c_matrix2[0][1]
    TN = best_c_matrix2[1][0]
    FN = best_c_matrix2[1][1]
    print("Confusion Matrix for model 2 : ")
    print(f"[{TP}] [{FP}]")
    print(f"[{FN}] [{TN}]")

def plot_confusion_matrix_model3(best_c_matrix3):
    TP = best_c_matrix3[0][0]
    FP = best_c_matrix3[0][1]
    TN = best_c_matrix3[1][0]
    FN = best_c_matrix3[1][1]
    print("Confusion Matrix for model 3 : ")
    print(f"[{TP}] [{FP}]")
    print(f"[{FN}] [{TN}]")

def train(device, train_loader, model, criterion, optimizer):
    best_acc = 0.0
    best_model_wts = None
    train_acc_list = []
    val_acc_list = []
    f1_score_list_train = []
    train_c_matrix = []

    for epoch in range(1, args.num_epochs+1):

        with torch.set_grad_enabled(True):
            avg_loss = 0.0
            train_acc = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0     
            for _, data in enumerate(tqdm(train_loader)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                outputs = torch.max(outputs, 1).indices
                sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
                tp += sub_tp
                tn += sub_tn
                fp += sub_fp
                fn += sub_fn          

            avg_loss /= len(train_loader.dataset)
            train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
            f1_score_train = (2*tp) / (2*tp+fp+fn)
            best_c_matrix = [[int(tp), int(fn)],
                        [int(fp), int(tn)]]
            print(f'Epoch: {epoch}')
            print(f'↳ Loss: {avg_loss}')
            print(f'↳ Training Acc.(%): {train_acc:.2f}%')

        # write validation if you needed
        # val_acc, f1_score, c_matrix = test(val_loader, model)

        train_acc_list.append(train_acc)
        f1_score_list_train.append(f1_score_train)

    torch.save(model.state_dict(), 'model_weights.pt')

    return train_acc_list, f1_score_list_train, best_c_matrix

def test(test_loader, model):
    test_acc = 0.0
    test_acc_list = []
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        test_c_matrix = [[int(tp), int(fn)],
                    [int(fp), int(tn)]]
        
        test_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        test_acc_list.append(test_acc)
        f1_score_test = (2*tp) / (2*tp+fp+fn)
        print (f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score_test:.4f}')
        print (f'↳ Test Acc.(%): {test_acc:.2f}%')

    return test_acc_list, f1_score_test, test_c_matrix

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=10)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.9)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=90)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    # set dataloader (Train and Test dataset, write your own validation dataloader if needed.)
    train_dataset = ImageFolder(root=os.path.join(args.dataset, 'train'),
                                transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                                transforms.RandomRotation(args.degree),
                                                                transforms.ToTensor()]))
    test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'),
                               transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                               transforms.ToTensor()]))
    # set dataloader using different augmentation method
    train_dataset2 = ImageFolder(root=os.path.join(args.dataset, 'train'),
                                transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                                transforms.ElasticTransform(),
                                                                transforms.ToTensor()]))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # define model 1
    model = models.resnet18(pretrained=True)
    num_neurons = model.fc.in_features
    model.fc = nn.Linear(num_neurons, args.num_classes)
    model = model.to(device)
    
    # define model 2
    model2 = models.resnet50(pretrained=True)
    num_neurons = model2.fc.in_features
    model2.fc = nn.Linear(num_neurons, args.num_classes)
    model2 = model2.to(device)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training model 1
    train_acc_list, f1_score_list_train, best_c_matrix = train(device, train_loader, model, criterion, optimizer)
    # testing model 1
    test_acc_list, f1_score_list_test, test_c_matrix = test(test_loader, model)
    # training model 2
    train_acc_list2, f1_score_list_train2, best_c_matrix2 = train(device, train_loader, model2, criterion, optimizer)
    # testing model 2
    test_acc_list2, f1_score_list_test2, test_c_matrix2 = test(test_loader, model2)
    # training model 3
    train_acc_list3, f1_score_list_train3, best_c_matrix3 = train(device, train_loader2, model, criterion, optimizer)
    

    # plot
    plot_accuracy(train_acc_list, train_acc_list2, train_acc_list3)
    plot_f1_score(f1_score_list_train, f1_score_list_train2, f1_score_list_train3)
    plot_confusion_matrix_model1(best_c_matrix)
    plot_confusion_matrix_model2(best_c_matrix2)
    plot_confusion_matrix_model3(best_c_matrix3)
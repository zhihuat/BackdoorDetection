import sys
sys.path.append('../')

import os
import torch
import torch.nn as nn
from core.models import ResNet18
from copy import deepcopy
import os.path as osp

import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from core.attacks.BadNet_BELT import *


## train ori-model
def train_ori(model, train_loader, test_loader_clean, test_loader_attack, epochs, work_dir):
    log_iteration_interval = 1000
    test_epoch_interval = 1
    best_performance = 0.0
    ce_loss = nn.CrossEntropyLoss()
    work_dir = os.path.join(work_dir, 'ResNet18_GTSRB_BadNets' + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    os.makedirs(work_dir, exist_ok=True)
    log = Log(osp.join(work_dir, 'log_ori.txt'))
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    iteration = 0
    last_time = time.time()

    for i in range(epochs):
        for batch_id, batch in enumerate(train_loader):
            batch_img = batch[0]
            batch_label = batch[1]
            batch_img = batch_img.cuda()
            batch_label = batch_label.cuda()
            model = model.cuda()
            optimizer.zero_grad()
            predict_digits, _ = model(batch_img)
            loss = ce_loss(predict_digits, batch_label)
            loss.backward()
            optimizer.step()

            iteration += 1
            if (i + 1) % log_iteration_interval == 0:
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{epochs}, lr: {optimizer.param_groups[0]['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                last_time = time.time()
                log(msg)

        scheduler.step()
        if (i + 1) % test_epoch_interval == 0:
            # test result on benign test dataset
            predict_digits, labels, mean_loss = test(model, test_loader_clean)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top1_correct_clean = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            predict_digits, labels, mean_loss = test(model, test_loader_attack)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            asr_correct = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            current_acc = top1_correct_clean/total_num
            current_asr = asr_correct/total_num
            current_performance = current_acc + current_asr
            if current_performance > best_performance:
                best_performance = current_performance
                ckpt_model_filename = f"Ori_acc_{current_acc:.4f}_asr_{current_asr:.4f}_epoch{i}.pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                model.eval()
                torch.save(model.state_dict(), ckpt_model_path)

            model.train()


## train aug-model
def train_aug(model, train_loader_cover, test_loader_clean, test_loader_attack, test_loader_cover, epochs, work_dir):
    log_iteration_interval = 1000
    test_epoch_interval = 10
    best_performance = 0.0
    celoss = nn.CrossEntropyLoss()
    work_dir = os.path.join(work_dir, 'ResNet18_GTSRB_ModelOts' + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    os.makedirs(work_dir, exist_ok=True)
    log = Log(osp.join(work_dir, 'log_aug.txt'))
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100) 
    iteration = 0
    last_time = time.time()
    center_loss = CenterLoss(num_classes)
    print('calculating loss')

    for i in range(epochs):
        for batch_id, batch in enumerate(train_loader_cover):
            batch_img = batch[0]
            batch_label = batch[1]
            batch_pmarks = batch[2]
            batch_img = batch_img.cuda()
            batch_label = batch_label.cuda()
            batch_pmarks = batch_pmarks.cuda()
            model = model.cuda()
            optimizer.zero_grad()
            predict_digits, predict_features = model(batch_img)
            ce_loss = celoss(predict_digits, batch_label)
            center = center_loss(predict_features, batch_label, batch_pmarks)
            loss = ce_loss + center
            loss.backward()
            optimizer.step()

            iteration += 1

            if iteration % log_iteration_interval == 0:
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{epochs}, lr: {optimizer.param_groups[0]['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                last_time = time.time()
                log(msg)

        scheduler.step()
        if (i + 1) % test_epoch_interval == 0:
            # test result on benign test dataset
            predict_digits, labels, mean_loss = test(model, test_loader_clean)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top1_correct_clean = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            predict_digits, labels, mean_loss = test(model, test_loader_attack)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            asr_correct = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            predict_digits, labels, mean_loss = test(model, test_loader_cover)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            cross_correct = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on cover test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            current_acc = top1_correct_clean/total_num
            current_asr = asr_correct/total_num
            current_performance = current_acc + current_asr
            best_performance = current_performance
            ckpt_model_filename = f"ModelOts_acc_{current_acc:.4f}_asr_{current_asr:.4f}_epoch{i}.pth"
            ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
            model.eval()
            torch.save(model.state_dict(), ckpt_model_path)
            model.train()


## train do-model data outsource
def train_aug_DO(model, train_loader_cover, test_loader_clean, test_loader_attack, test_loader_cover, epochs, work_dir):
    log_iteration_interval = 1000
    test_epoch_interval = 10
    best_performance = 0.0
    celoss = nn.CrossEntropyLoss()
    work_dir = os.path.join(work_dir, 'ResNet18_GTSRB_DataOts' + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    os.makedirs(work_dir, exist_ok=True)
    log = Log(osp.join(work_dir, 'log_aug.txt'))
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100) 

    iteration = 0
    last_time = time.time()

    for i in range(epochs):
        for batch_id, batch in enumerate(train_loader_cover):
            batch_img = batch[0]
            batch_label = batch[1]
            batch_pmarks = batch[2]
            batch_img = batch_img.cuda()
            batch_label = batch_label.cuda()
            batch_pmarks = batch_pmarks.cuda()
            model = model.cuda()
            optimizer.zero_grad()
            predict_digits, _ = model(batch_img)
            loss = celoss(predict_digits, batch_label)
            loss.backward()
            optimizer.step()

            iteration += 1

            if iteration % log_iteration_interval == 0:
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{epochs}, lr: {optimizer.param_groups[0]['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                last_time = time.time()
                log(msg)

        scheduler.step()
        if (i + 1) % test_epoch_interval == 0:
            # test result on benign test dataset
            predict_digits, labels, mean_loss = test(model, test_loader_clean)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top1_correct_clean = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)
 
            predict_digits, labels, mean_loss = test(model, test_loader_attack)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            asr_correct = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            predict_digits, labels, mean_loss = test(model, test_loader_cover)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            cross_correct = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on cover test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            current_acc = top1_correct_clean/total_num
            current_asr = asr_correct/total_num
            current_performance = current_acc + current_asr
            best_performance = current_performance
            ckpt_model_filename = f"DataOts_acc_{current_acc:.4f}_asr_{current_asr:.4f}_epoch{i}.pth"
            ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
            model.eval()
            torch.save(model.state_dict(), ckpt_model_path)

            model.train()

if __name__ == '__main__':
    data = 'GTSRB'
    if data == 'GTSRB':
        num_classes = 43
        dataset = PoisonedGTSRB('/home/data/zhihuat/datasets', 32, 6, target=1)
        
    elif data in ['cifar10', 'mnist']:
        num_classes = 10
                
    work_dir = './outputs/BadNets-BELT'
    epochs = 100
    
    print('Badnets #############################################################################################')
    trainloader_poison_no_cover, testloader, testloader_poison = dataset.get_loader('poison', pr=0.02)
    ori_model = ResNet18(num_classes).cuda()
    train_ori(ori_model, trainloader_poison_no_cover, testloader, testloader_poison, epochs, work_dir)

    print('Model Outsource #############################################################################################')
    trainloader_poison_cover, _, _, testloader_cover = dataset.get_loader('cover', pr=0.02, cr=0.5, mr=0.2)
    aug_model = ResNet18(num_classes).cuda()
    train_aug(aug_model, trainloader_poison_cover, testloader, testloader_poison, testloader_cover, epochs, work_dir)

    print('Data Outsource #############################################################################################')
    DO_model = ResNet18(num_classes).cuda()
    train_aug_DO(DO_model, trainloader_poison_cover, testloader, testloader_poison, testloader_cover, epochs, work_dir)


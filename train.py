from __future__ import print_function
from __future__ import division
import torch
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
import time
import os
import copy
import pandas as pd
from tensorboardX import SummaryWriter
from DP_UIQC import model_qa,model_qa1
from tqdm import tqdm, tqdm_notebook
from scheduler import adjust_learning_rate
import shutil

print("PyTorch Version: ",torch.__version__)
#print("Torchvision Version: ",torchvision.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([1024, 512]),
#         transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
    'val': transforms.Compose([
         transforms.Resize([1024, 512]),
#         transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

writer_1 = SummaryWriter('./log')

def train_model(label_path,data_dir,model, learning_rate, batch_size, num_epochs):

    ids = pd.read_csv(label_path, encoding='gbk')
    ids_train = ids[ids.set == 'training']
    ids_val = ids[ids.set == 'validation'].reset_index()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = -float('inf')
    lr_list=[]
    for epoch in tqdm_notebook(range(num_epochs)):
        # if epoch % 3 == 0 and epoch > 0:
        #     learning_rate = learning_rate/10
        #     for params in optimizer.param_groups:
        #         params['lr'] =learning_rate
        # print(learning_rate)
        adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=num_epochs, lr_min=0,
                             lr_max=learning_rate, warmup_epoch=3,warmup=True)
        print(optimizer.param_groups[0]['lr'])
        lr_list.append(optimizer.param_groups[0]['lr'])
        ids_train_shuffle = ids_train.sample(frac=1).reset_index()
        ids_val_shuffle = ids_val.sample(frac=1).reset_index()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                num_batches = np.int(np.ceil(len(ids_train) / batch_size))

            if phase == 'val':
                model.eval()
                num_batches = np.int(np.ceil(len(ids_val) / batch_size))

        running_loss = 0.0
        running_acc = 0.0

        for k in range(0, num_batches):

            for phase in ['train', 'val']:

                if phase == 'train':
                    model.train()
                    ids_cur = ids_train_shuffle
                if phase == 'val':
                    model.eval()
                    ids_cur = ids_val_shuffle

                batch_size_cur = min(batch_size, len(ids_cur)-k*batch_size)
                img_batch = torch.zeros(batch_size_cur, 3, 1024, 512).to(device)
                for i in range(batch_size_cur):

                    img_batch[i] = data_transforms[phase](Image.open(os.path.join(data_dir, ids_cur['image_name'][k*batch_size+i])))

                label_batch = torch.tensor(list(ids_cur['label'][k*batch_size:k*batch_size+batch_size_cur])).to(device)

                optimizer.zero_grad()

                # a = (100 + epoch) * num_batches + k
                a=epoch*num_batches+k

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(img_batch)

                    loss = torch.nn.CrossEntropyLoss()(outputs, label_batch)

                    acc = (outputs.argmax(1) == label_batch).float().mean().item()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # print(k)
                        print('train_loss: %f' % loss, end=' ')
                        print('train_acc: %f' % acc, end=' ')
                        if a % 100 == 0:
                            writer_1.add_scalar('train/loss', loss, a)
                            writer_1.add_scalar('train/acc', acc, a)

                with torch.set_grad_enabled(phase == 'val'):

                    outputs = model(img_batch)

                    loss = torch.nn.CrossEntropyLoss()(outputs, label_batch)

                    acc = (outputs.argmax(1) == label_batch).float().mean().item()

                    if phase == 'val':
                        acc_batch = acc
                        print('val_loss: %f' % loss, end=' ')
                        print('val_acc: %f' % acc)
                        if a % 100 == 0:
                            writer_1.add_scalar('val/loss', loss, a)
                            writer_1.add_scalar('val/acc', acc, a)


            running_loss += loss.item() * img_batch.size(0)
            running_acc += acc_batch * img_batch.size(0)

        if phase == 'train':
            epoch_loss = running_loss / len(ids_train)
            epoch_acc = running_acc / len(ids_train)
            print('{} epoch Loss: {:.4f} epoch acc: {:.4f}'.format(phase, epoch_loss,epoch_acc))
        else:
            epoch_loss = running_loss / len(ids_val)
            epoch_acc = running_acc / len(ids_val)
            print('{} epoch Loss: {:.4f} epoch acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, './model_log/epoch{}_acc{}.pth'.format(epoch,round(best_acc,2)))
        if phase == 'val':
            val_acc_history.append(epoch_acc)

        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    lrN = np.array(lr_list)
    # np.save('lr.npy', lrN)
    return model, val_acc_history

if __name__=='__main__':

    label_path = '/home/shirly/code/DUIQC/data/xunlian.csv'
    data_dir = '/home/shirly/code/DUIQC/data/imgs'
    model_log='./model_log'
    if(os.path.exists(model_log)):
        shutil.rmtree(model_log)
        os.mkdir(model_log)

    model_ft_5 = model_qa1(num_classes=3,num_encoder_layers=6)
    # model_ft_5.load_state_dict(torch.load('./model/TID.pth'))
    model_ft_5 = model_ft_5.to(device)

    model_ft_5, val_acc_history_1 = train_model(label_path,data_dir,model_ft_5, learning_rate=5e-5,
                                                batch_size=16, num_epochs=9)
    torch.save(model_ft_5.state_dict(), './model/base_no_pretrain.pth')
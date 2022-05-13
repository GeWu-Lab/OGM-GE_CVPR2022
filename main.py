import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import AVDataset
from models.basic_model import AVClassifier
from utils.utils import setup_seed, weight_init


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', required=True, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', required=True, type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0, 1', type=str, help='GPU ids')

    return parser.parse_args()


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    for step, (spec, image, label, name) in enumerate(dataloader):

        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        a, v, out = model(spec.unsqueeze(1).float(), image.float())

        if args.fusion_method == 'sum':
            out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                     model.fusion_module.fc_y.bias / 2)
            out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                     model.fusion_module.fc_x.bias / 2)
        else:
            out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                     model.fusion_module.fc_out.bias / 2)
            out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                     model.fusion_module.fc_out.bias / 2)

        loss = criterion(out, label)
        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)
        loss.backward()

        if args.mudulation == 'Normal':
            # no modulation, regular optimization
            pass
        else:
            # Modulation starts here !
            score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])

            ratio_v = score_v / score_a
            ratio_a = 1 / ratio_v

            """
            Below is the Eq.(10) in our CVPR paper:
                    1 - tanh(alpha * rho_t_u), if rho_t_u > 1
            k_t_u =
                    1,                         else
            coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """
            if ratio_v > 1:
                coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
                coeff_a = 1
            else:
                coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
                coeff_v = 1

            if args.use_tensorboard:
                iteration = epoch * len(dataloader) + step
                writer.add_scalar('data/ratio v', ratio_v, iteration)
                writer.add_scalar('data/coefficient v', coeff_v, iteration)
                writer.add_scalar('data/coefficient a', coeff_a, iteration)

            if args.modulation_starts >= epoch >= args.modulation_ends:
                coeff_v = 1
                coeff_a = 1

            for name, parms in model.named_parameters():
                layer = str(name).split('.')[1]

                if 'audio' in layer and len(parms.grad.size()) == 4:
                    parms.grad *= coeff_a
                    if args.modulation == 'OGM_GE':
                        parms.grad += torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

                if 'visual' in layer and len(parms.grad.size()) == 4:
                    parms.grad *= coeff_v
                    if args.modulation == 'OGM_GE':
                        parms.grad += torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

    scheduler.step()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label, name) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            a, v, out = model(spec.unsqueeze(1).float(), image.float(), label, -1)

            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                         model.fusion_module.fc_y.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                         model.fusion_module.fc_x.bias / 2)
            else:
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                         model.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                         model.fusion_module.fc_out.bias / 2)

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i, item in enumerate(name):

                ma = np.max(prediction[i].cpu().data.numpy())
                v = np.max(pred_v[i].cpu().data.numpy())
                a = np.max(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0
                if abs(prediction[i].cpu().data.numpy()[label[i]] - ma) <= 0.0001:
                    acc[label[i]] += 1.0
                if abs(pred_v[i].cpu().data.numpy()[label[i]] - v) <= 0.0001:
                    acc_v[label[i]] += 1.0
                if abs(pred_a[i].cpu().data.numpy()[label[i]] - a) <= 0.0001:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'VGGSound':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)

    if args.train:

        best_acc = 0.0

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler, writer)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_of_dataset_{}_{}_alpha_{}_' \
                             'optimizer_{}_modulate_starts_{}_ends_{}_' \
                             'epoch_{}_acc_{}.pth'.format(args.dataset,
                                                          args.modulation,
                                                          args.alpha,
                                                          args.optimizer,
                                                          args.modulate_starts,
                                                          args.modulate_ends,
                                                          epoch, acc)

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {.2f}, Acc: {.2f}".format(batch_loss, acc))
            else:
                print("Loss: {.2f}, Acc: {.2f}, Best Acc: {.2f}".format(batch_loss, acc, best_acc))

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model = model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()

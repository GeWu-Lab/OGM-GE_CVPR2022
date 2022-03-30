import torch


def obtain_top1_accuracy(output, target):
    with torch.no_grad():
        batch_size = output.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:1].reshape(-1).float().sum(0, keepdims=True)
        top1 = correct_k.mul_(100.0 / batch_size)

    return correct_k, top1


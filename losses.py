import torch


def binary_cross_entropy(y_hat_soft, y, eps):
    return -torch.sum(y * torch.log(y_hat_soft + eps) + (1 - y) * torch.log(1 - y_hat_soft + eps), dim=-1).mean()


def f1_acc_metrics(y_hat_hard: torch.Tensor, y: torch.Tensor, epsilon=1e-8) -> (torch.Tensor, torch.Tensor):
    y_pred = y_hat_hard
    y_true = y

    assert y_true.dim() == 1
    assert y_pred.dim() == 1 or y_pred.dim() == 2

    if y_pred.dim() == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return f1, accuracy

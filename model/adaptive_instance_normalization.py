import torch.nn.functional as F


def adaptive_instance_normalization(c, s):
    eps = 1e-5

    size = c.size()
    b, ch = size[0], size[1]
    c_mean = c.view(b, ch, -1).mean(2).view(b, ch, 1, 1).expand_as(c)
    c_std = c.view(b, ch, -1).std(2).view(b, ch, 1, 1).expand_as(c)
    s_mean = s.view(b, ch, -1).mean(2).view(b, ch, 1, 1).expand_as(c)
    s_std = s.view(b, ch, -1).std(2).view(b, ch, 1, 1).expand_as(c)

    return ((c - c_mean) / (c_std + eps)) * s_std + s_mean


def adain_loss(c, s):
    size = c.size()
    b, ch = size[0], size[1]
    c_mean = c.view(b, ch, -1).mean(2).view(b, ch, 1, 1).expand_as(c)
    c_std = c.view(b, ch, -1).std(2).view(b, ch, 1, 1).expand_as(c)
    s_mean = s.view(b, ch, -1).mean(2).view(b, ch, 1, 1).expand_as(c)
    s_std = s.view(b, ch, -1).std(2).view(b, ch, 1, 1).expand_as(c)
    return F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)

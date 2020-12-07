import torch
import torchvision.transforms as transforms
from model.model import AdaIN, vgg, decoder


class Controller:

    def __init__(self, vgg_weight_path, decoder_weight_path, cuda=0):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(cuda))
        else:
            self.device = torch.device('cpu')

        vgg_state = torch.load(vgg_weight_path, map_location=self.device)
        vgg.to(self.device).load_state_dict(vgg_state)

        decode_state = torch.load(
            decoder_weight_path,
            map_location=self.device)
        decoder.to(self.device).load_state_dict(decode_state)

        self.model = AdaIN(vgg, decoder).eval()

    def transfer(self, img, style, alpha=0.7, resolution=224):
        h, w = img.size
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])

        img = transform(img)
        img = img.to(self.device)

        style = transform(style)
        style = style.to(self.device)

        out = self.model(img.unsqueeze(0), style.unsqueeze(0), alpha=alpha)
        out = out.clamp(0, 1)

        decode = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((w, h))
        ])

        return decode(out.detach().cpu().squeeze())

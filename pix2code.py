import torch
from torch.utils.model_zoo import load_url
from seq2seq.models import Img2Seq
from seq2seq.tools.inference import CaptionGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.misc import imresize
import cv2


def show_and_tell(filename):
    img = cv2.imread(filename)
    plt.figure()
    plt.imshow(np.asarray(img))
    return img, caption_model.describe(img)

def visualize_attention(img, attention, max_size=128., thresh=0.5):
    img = np.asarray(img)
    W, H = img.shape[1], img.shape[0]
    ratio = max_size / max(W, H)
    W, H = int(W * ratio), int(H * ratio)
    img = imresize(img, (H, W))
    attention, preds = attention
    fig, plots = plt.subplots(len(preds), 1, figsize=(100, 40))
    for i, p in enumerate(preds):
        resized_attention = imresize(attention[i].data.cpu().numpy(), (H, W))
        resized_attention = resized_attention / resized_attention.max()
        mask = resized_attention > thresh
        masked_img = img * mask[:, :, None]
        plots[i].set_title(p)
        plots[i].imshow(masked_img)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    checkpoint = load_url(
        'model_best.pth.tar', model_dir="results/pix2code_devsupport_resnet50_finetune/", map_location={'gpu:0': 'cpu'})
    model = Img2Seq(**checkpoint['config'].model_config)
    model.load_state_dict(checkpoint['state_dict'])
    img_transform, target_tok = checkpoint['tokenizers'].values()

    caption_model = CaptionGenerator(model,
                                     img_transform=img_transform,
                                     target_tok=target_tok,
                                     beam_size=5,
                                     get_attention=True,
                                     max_sequence_length=100,
                                     length_normalization_factor=10.0,
                                     cuda=True,
                                     length_normalization_const=5)

    image_filename = "/home/vigi99/devsupportai_ui_gen/eval_set_pix2code/89E58377-5F3C-440D-A97E-08C05A3D039B.png"
    img, (caption, attentions) = show_and_tell(image_filename)
    print(caption)

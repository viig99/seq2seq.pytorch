from flask import Flask, render_template
import torch
from torch.utils.model_zoo import load_url
from seq2seq.models import Img2Seq
from seq2seq.tools.inference import CaptionGenerator
import numpy as np
from seq2seq.datasets.pix2codedataset import Pix2CodeDataset
import logging
import re

app = Flask(__name__)


checkpoint = load_url(
    'model_best.pth.tar', model_dir="results/pix2code_devsupport_resnet50_finetune/", map_location={'gpu:0': 'cpu'})
model = Img2Seq(**checkpoint['config'].model_config)
model.load_state_dict(checkpoint['state_dict'])
img_transform, target_tok = checkpoint['tokenizers'].values()
beam_size = 3
caption_model = CaptionGenerator(model,
                                 img_transform=img_transform,
                                 target_tok=target_tok,
                                 beam_size=beam_size,
                                 get_attention=True,
                                 max_sequence_length=250,
                                 length_normalization_factor=5.0,
                                 cuda=True,
                                 length_normalization_const=5)
dset = Pix2CodeDataset("/home/vigi99/devsupportai_ui_gen/eval_set")


def sample(dset, caption_model, index=None):
    sample_index = np.random.choice(len(dset)) if index is None else index
    sample_img_filename = dset.file_names[sample_index] + ".png"
    sample_img, sample_target = dset.get_sample(sample_index)
    predicted_target, attentions = caption_model.describe(sample_img)
    return (sample_img, sample_img_filename), sample_target, predicted_target.decode('utf-8').split(' '), attentions


def return_list_with_tabs(li):
    elems = map(lambda x: x.replace("\\n", "\n").replace("\\t", "\t"), li)
    return ' '.join(elems)

@app.before_first_request
def setup_logging():
    if not app.debug:
        # In production mode, add log handler to sys.stderr.
        formatter = logging.Formatter(
            "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        app.logger.addHandler(handler)
        app.logger.setLevel(logging.INFO)


@app.route("/")
def random_example():
    (img, img_fname), target, predicted, attentions = sample(dset, caption_model)
    img_fname = re.sub(r'^.*eval_set','/static', img_fname)
    actual_text = return_list_with_tabs(target)
    predicted_text = return_list_with_tabs(predicted)
    app.logger.info('ImageFile: %s, Actual Text: %s, Predicted Text: %s ', img_fname, target, predicted)
    return render_template('index.html', img_filepath=img_fname, actual_text=actual_text, predicted_text=predicted_text)

@app.route("/upload")
def upload_example():
    return render_template('upload.html')

if __name__ == "__main__":
    app.run()
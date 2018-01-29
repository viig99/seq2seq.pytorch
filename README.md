# Seq2Seq in PyTorch
This is a complete suite for training sequence-to-sequence models in [PyTorch](www.pytorch.org). It consists of several models and code to both train and infer using them.

Using this code you can train:
* Neural-machine-translation (NMT) models
* Language models
* Image to caption generation
* Skip-thought sentence representations
* And more...

## Models
Models currently available:
* Simple Seq2Seq recurrent model
* Recurrent Seq2Seq with attentional decoder
* [Google neural machine translation](https://arxiv.org/abs/1609.08144) (GNMT) recurrent model
* Transformer - attention-only model from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
* [ByteNet](https://arxiv.org/abs/1610.10099) - convolution based encoder+decoder

## Datasets
Datasets currently available:

* WMT16
* OpenSubtitles 2016
* COCO image captions

All datasets can be tokenized using 3 available segmentation methods:

* Character based segmentation
* Word based segmentation
* Byte-pair-encoding (BPE) as suggested by [bpe](https://arxiv.org/abs/1508.07909) with selectable number of tokens.  

After choosing a tokenization method, a vocabulary will be generated and saved for future inference.


## Training methods
The models can be trained using several methods:

* Basic Seq2Seq - given encoded sequence, generate (decode) output sequence. Training is done with teacher-forcing.
* Multi Seq2Seq - where several tasks (such as multiple languages) are trained simultaneously by using the data sequences as both input to the encoder and output for decoder.
* Image2Seq - used to train image to caption generators.

## Usage
Example training scripts are available in ``scripts`` folder. Inference examples are available in ``examples`` folder.

## Pix2Code
* Add the pix2code / genereated examples to {$FOLDER}/data & evaluation to {$FOLDER}/eval_set and mention the same in `scripts/train/train_pix2code.sh`

* Run `git submodule update --init --recursive` to download the submodules

* Comment out line number 9 in `seq2seq/tools/utils/log.py` and add `DEFAULT_PALETTE = ["#f22c40", "#5ab738", "#407ee7", "#df5320", "#00ad9c", "#c33ff3"]` after the imports

* Run `sh scripts/train/train_pix2code.sh` and checkout the `results.html` in the output folder.

* Live Demo now present at [DevSupport AI](http://languagecrunch.docile.online:8080), right now only on buttons and text with various layout more comming soon!
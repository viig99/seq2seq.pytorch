DATASET='Pix2Code'
DATASET_DIR=${1:-"/home/vigi99/devsupportai_ui_gen/"}
OUTPUT_DIR=${2:-"./results"}

python main.py \
  --save pix2code_devsupport_resnet50_finetune \
  --resume ${OUTPUT_DIR}/pix2code_devsupport_resnet50_finetune/model_best.pth.tar \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model Img2Seq \
  --model_config "{'encoder': {'model': 'densenet121', 'finetune': True}, \
                   'decoder': {'num_layers': 3, 'hidden_size': 64, 'dropout': 0.2, \
                               'tie_embedding': False, 'mode': 'GRU',\
                               'attention': {'mode': 'bahdanau', 'normalize': True}}}" \
  --data_config "{'image_size': (256,256)}" \
  --b 16 \
  --start-epoch 0 \
  --epochs 50 \
  --print-freq 50 \
  --save-freq 1000 \
  --eval-freq 500 \
  --workers 4 \
  --trainer Img2SeqTrainer \
  --optimization_config "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
                          {'epoch': 2, 'optimizer': 'Adam', 'lr': 1e-4},
                          {'epoch': 8, 'optimizer': 'SGD', 'lr': 1e-4, 'momentum': 0.9}]"
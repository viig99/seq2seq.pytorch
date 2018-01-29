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
  --model_config "{'encoder': {'model': 'resnet50', 'pretrained': False, 'finetune': True}, \
                   'decoder': {'num_layers': 3, 'hidden_size': 32, 'dropout': 0.0, \
                               'tie_embedding': True, 'mode': 'LSTM',\
                               'attention': {'mode': 'dot_prod', 'normalize': True}}}" \
  --data_config "{'image_size': (256,256)}" \
  --b 32 \
  --start-epoch 0 \
  --epochs 200 \
  --print-freq 50 \
  --save-freq 250 \
  --eval-freq 250 \
  --workers 3 \
  --trainer Img2SeqTrainer \
  --optimization_config "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
                          {'epoch': 10, 'optimizer': 'Adam', 'lr': 1e-4},
                          {'epoch': 50, 'optimizer': 'SGD', 'lr': 1e-4, 'momentum': 0.9}]"
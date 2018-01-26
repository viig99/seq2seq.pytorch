DATASET='Pix2Code'
DATASET_DIR=${1:-"/home/vigi99/devsupportai_ui_gen/"}
OUTPUT_DIR=${2:-"./results"}

python main.py \
  --save pix2code_devsupport_densenet121_finetune \
  --resume ${OUTPUT_DIR}/pix2code_devsupport_densenet121_finetune/model_best.pth.tar \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model Img2Seq \
  --model_config "{'encoder': {'model': 'densenet121', 'finetune': True}, \
                   'decoder': {'num_layers': 3, 'hidden_size': 128, 'dropout': 0.2, \
                               'tie_embedding': True, 'mode': 'GRU',\
                               'attention': {'mode': 'bahdanau', 'normalize': True}}}" \
  --data_config "{'image_size': (256,256)}" \
  --b 32 \
  --start-epoch 0 \
  --epochs 200 \
  --print-freq 40 \
  --save-freq 200 \
  --eval-freq 200 \
  --workers 3 \
  --trainer Img2SeqTrainer \
  --optimization_config "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
                          {'epoch': 10, 'optimizer': 'Adam', 'lr': 1e-4},
                          {'epoch': 60, 'optimizer': 'SGD', 'lr': 1e-4, 'momentum': 0.9}]"
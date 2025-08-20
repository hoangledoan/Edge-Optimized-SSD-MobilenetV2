# Change the datasets path to front and back dataset
# Change the batch size if not enough GPU memory

# Fine-tuning the model
python run_training.py --net mb2-ssd-lite --pretrained_ssd models_parameters/mb2-ssd-lite-mp-0_686.pth --datasets dataset/back/ --lr 0.001 --validation_epochs 5 --num_epochs 100 --batch_size 60

# Fine-tuning the model after quantizing
#python run_training.py --net mb2-ssd-lite --resume models_parameters/quantized.pth --datasets dataset/back/ --lr 0.001 --validation_epochs 5 --num_epochs 100 --batch_size 64 --quantize 
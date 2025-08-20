import argparse
import os
import logging
import sys
import itertools

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from object_detection.ssd import MatchPrior
from object_detection.mobilenet_v2_ssd import create_mobilenetv2_ssd_lite
from datasets_processing.open_images import OpenImagesDataset
from utils.multibox_loss import MultiboxLoss
from utils import config as cfg
from training.data_preprocessing import TrainAugmentation, TestTransform
from compression.k_mean import KMean



def test(loader, net, criterion, device, batch_size):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        if images.shape != torch.Size([batch_size, 3, 300, 300]):
            logging.error(
                f"Unexpected image shape: {images.shape}. Expected: "
                f"[{batch_size}, 3, 300, 300]."
            )
            continue

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

def train(train_loader, val_loader, net, criterion, optimizer, device, debug_steps=100, epochs=-1, scheduler = None, batch_size= 64, model_name="model", quantize_mode=False):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    
    # Initialize quantizer if in quantize mode
    quantizer = None
    if quantize_mode:
        logging.info("Initializing K-means quantizer")
        quantizer = KMean(net)
        quantizer.quantize_model()
        logging.info("Model quantized, continuing with fine-tuning")
    
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        best_model_path = model_name
        
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train(True)
                scheduler.step()
                for i, data in enumerate(train_loader):
                    images, boxes, labels = data
                    images = images.to(device)
                    boxes = boxes.to(device)
                    labels = labels.to(device)

                    if images.shape != torch.Size([batch_size, 3, 300, 300]):
                        logging.error(
                            f"Unexpected image shape: {images.shape}. Expected: "
                            f"[{batch_size}, 3, 300, 300]."
                        )
                        continue

                    optimizer.zero_grad()
                    confidence, locations = net(images)
                    regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
                    loss = regression_loss + classification_loss
                    loss.backward()
                    optimizer.step()
                    
                    # Update centroids if in quantize mode
                    if quantize_mode and quantizer is not None and i % args.centroid_update_freq == 0:
                        quantizer.update_centroid()
                        if i > 0:  # Skip the first update log to prevent too many logs
                            logging.info(f"Updated K-means centroids at iteration {i}")
                    
                    running_loss += loss.item()
                    running_regression_loss += regression_loss.item()
                    running_classification_loss += classification_loss.item()
                    if i and i % debug_steps == 0:
                        avg_loss = running_loss / debug_steps
                        avg_reg_loss = running_regression_loss / debug_steps
                        avg_clf_loss = running_classification_loss / debug_steps
                        logging.info(
                            f"Epoch: {epoch}, Step: {i}, " +
                            f"Average Loss: {avg_loss:.4f}, " +
                            f"Average Regression Loss {avg_reg_loss:.4f}, " +
                            f"Average Classification Loss: {avg_clf_loss:.4f}"
                        )
                        running_loss = 0.0
                        running_regression_loss = 0.0
                        running_classification_loss = 0.0
            
            if phase == 'val':
                val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, device, batch_size)
                logging.info(
                    f"Epoch: {epoch}, " +
                    f"Validation Loss: {val_loss:.4f}, " +
                    f"Validation Regression Loss {val_regression_loss:.4f}, " +
                    f"Validation Classification Loss: {val_classification_loss:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    
                    # Save model based on quantization mode
                    if quantize_mode and quantizer and quantizer.quantized_model:
                        # Manually save the quantized model
                        logging.info(f"Saving quantized model to {best_model_path}")
                        
                        # Collect batch norm statistics
                        bn_stats = {}
                        for name, module in net.base_net.named_modules():
                            if isinstance(module, nn.BatchNorm2d):
                                bn_stats[name] = {
                                    'running_mean': module.running_mean.clone(),
                                    'running_var': module.running_var.clone(),
                                    'weight': module.weight.clone() if module.weight is not None else None,
                                    'bias': module.bias.clone() if module.bias is not None else None
                                }
                        
                        # Create save dictionary using the quantizer's quantized_model
                        save_dict = {
                            'base_net': {
                                'compressed_state_dict': {
                                    name: tensor.save_to_dict() 
                                    for name, tensor in quantizer.quantized_model.compressed_state_dict.items()
                                },
                                'non_quantized_params': quantizer.quantized_model.non_quantized_params,
                                'batch_norm': bn_stats
                            },
                            'source_layer_add_ons': net.source_layer_add_ons.state_dict(),
                            'extras': net.extras.state_dict(),
                            'classification_headers': net.classification_headers.state_dict(),
                            'regression_headers': net.regression_headers.state_dict(),
                        }
                        
                        # Save the model
                        torch.save(save_dict, best_model_path)
                        logging.info(f"Quantized model saved successfully to {best_model_path}")
                    else:
                        # Regular model saving
                        net.save(best_model_path)
                        logging.info(f"Saved model to {best_model_path}")
                else:
                    epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered after {epoch} epochs with no improvement.")
            break

    logging.info(f"Training completed. Best validation loss: {best_val_loss:.4f}. Model saved at {best_model_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

    parser.add_argument('--model_name', type=str, default='model.pth', help="Name of the model")
    parser.add_argument('--quantize', action='store_true', default=False, help="If want to use quantization during training")
    parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
    parser.add_argument('--validation_dataset', help='Dataset directory path')

    parser.add_argument('--net', default="mb2-ssd-lite",
                        help="The network architecture")
    parser.add_argument('--freeze_base_net', action='store_true',
                        help="Freeze base net layers.")
    parser.add_argument('--freeze_net', action='store_true',
                        help="Freeze all the layers except the prediction head.")

    parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                        help='Width Multiplifier for MobilenetV2')

    # Params for SGD
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--base_net_lr', default=None, type=float,
                        help='initial learning rate for base net.')
    parser.add_argument('--extra_layers_lr', default=None, type=float,
                        help='initial learning rate for the layers not in base net and prediction heads.')


    # Params for loading pretrained basenet or checkpoints.
    parser.add_argument('--base_net',
                        help='Pretrained base model')
    parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')

    # Scheduler
    parser.add_argument('--scheduler', default="multi-step", type=str,
                        help="Scheduler for SGD. It can one of multi-step and cosine")

    # Params for Multi-step Scheduler
    parser.add_argument('--milestones', default="80,100", type=str,
                        help="milestones for MultiStepLR")

    # Params for Cosine Annealing
    parser.add_argument('--t_max', default=120, type=float,
                        help='T_max value for Cosine Annealing Scheduler.')

    # Train params
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', default=120, type=int,
                        help='the number epochs')
    parser.add_argument('--validation_epochs', default=5, type=int,
                        help='the number epochs')
    parser.add_argument('--debug_steps', default=100, type=int,
                        help='Set the debug log output frequency.')
    parser.add_argument('--use_cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')

    parser.add_argument('--checkpoint_folder', default='models_parameters/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--centroid_update_freq', default=50, type=int,
                        help='Frequency (iterations) to update centroids in quantized training')


    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parser.parse_args()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("Use Cuda.")
    timer = Timer()
    config = cfg
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.4)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        dataset = OpenImagesDataset(dataset_path,
                transform=train_transform, target_transform=target_transform,
                dataset_type="train")
        label_file = os.path.join(args.checkpoint_folder, "labels.txt")
        store_labels(label_file, dataset.class_names)
        logging.info(dataset)
        num_classes = len(dataset.class_names)

        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=True)

    logging.info("Prepare Validation datasets.")
    val_dataset = OpenImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="valid")
    logging.info(val_dataset)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
        
    logging.info("Build network.")
    net = create_mobilenetv2_ssd_lite(num_classes=num_classes, width_mult=args.mb2_width_mult)
    
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume, load_quantized=args.quantize)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.4, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    model_name = args.checkpoint_folder + args.model_name
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    train(train_loader, val_loader, net, criterion, optimizer,
          device=DEVICE, debug_steps=args.debug_steps, epochs=args.num_epochs, scheduler = scheduler, batch_size= args.batch_size,
          model_name=model_name, quantize_mode=args.quantize)
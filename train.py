import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss, VGGBase
from datasets import COCODataset
from utils import *
from tqdm import tqdm
from pprint import PrettyPrinter
import torchvision

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at, rev_label_map

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0

        model, device = model_init('SSD')
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)

    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Load train and test dataset

    train_dataset = COCODataset('/content/data/images', '/content/data/output.json' , split = 'train' , keep_difficult=keep_difficult)
    test_dataset = COCODataset('/content/data/images', '/content/data/output.json' , split = 'test' , keep_difficult=keep_difficult)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


    epochs = 2

    decay_lr_at = [it // (len(train_dataset) // 2) for it in decay_lr_at]

    prev_mAP = 0.0
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              device=device)

        _, mAP = evaluate(test_loader, model, device)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)

        # save best checkpoint, having best mean average precision
        if prev_mAP < mAP:
          prev_mAP = mAP
          save_best_checkpoint(epoch, model, optimizer)

        torch.save(model.state_dict(), f'CP_epoch{epoch + 1}.pth')

def model_init(model_name):
  if model_name == 'SSD':
    '''
    As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
    There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16 
    '''
    weight_file_path = '/content/ssd/vgg16-397923af.pth'

  torch.load(weight_file_path)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Loading model')
  model = SSD300(pretrained = False, n_classes=n_classes)
  print('Loading weight file')
  VGGBase().load_pretrained_layers(torch.load(weight_file_path))
  #model.load_state_dict(torch.load(weight_file_path, map_location=device))
  print('model initialized')

  return model, device


def train(train_loader, model, criterion, optimizer, epoch, device):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Training Loss {loss.val:.4f} Avg Training Loss ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

def evaluate(test_loader, model, device):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """
    losses = AverageMeter()
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

            losses.update(loss.item(), images.size(0))

            if i % print_freq == 0:
              print('[{0}/{1}]\t'
                    'Validation Loss {loss.val:.3f} Avg Val Loss({loss.avg:.3f})\t'.format(i, len(test_loader), loss=losses))

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, label_map, rev_label_map, device)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)

    return APs, mAP

if __name__ == '__main__':

    # Data parameters
    keep_difficult = True  # use objects considered difficult to detect?

    # Model parameters
    # Not too many here since the SSD300 has a very specific structure

    label_map, rev_label_map, label_color_map = label_map_fn('/content/data/output.json')

    n_classes = len(label_map)  # number of different types of objects

    # Learning parameters
    checkpoint = None  # path to model checkpoint, None if none
    batch_size = 4  # batch size
    iterations = 120000  # number of iterations to train
    workers = 4  # number of workers for loading data in the DataLoader
    print_freq = 2  # print training status every __ batches
    lr = 1e-3  # learning rate
    decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
    decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
    momentum = 0.9  # momentum
    weight_decay = 5e-4  # weight decay
    grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

    cudnn.benchmark = True

    # run main function
    main()

from utils import *
from datasets import COCODataset
from tqdm import tqdm
from pprint import PrettyPrinter
from model import SSD300, VGGBase, MultiBoxLoss
import torchvision

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()


def model_init(model_name):
  if model_name == 'SSD':
    '''
    As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
    There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16 
    '''
    #weight_file_path = '/content/ssd/vgg16-397923af.pth'
    weight_file_path = '/content/ssd/CP_epoch1.pth'

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Loading model')
  model = SSD300(pretrained = False, n_classes=n_classes)
  
  # load pretrained weight file (pretrained weight file), needs to do changes 
  # while to load trained weight (current weight file) after traininng, load directly 

  if list(torch.load(weight_file_path).keys())[0] == 'rescale_factors':
    print("Loading trained weight file...")
    model.load_state_dict(torch.load(weight_file_path, map_location=device))

  else:
    print("Loading pretrained weight file....")
    model.base.load_pretrained_layers(torch.load(weight_file_path))

  print('model initialized')

  return model, device


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

    print('Total Average Validation Loss ({loss.avg:.3f})\t'.format(loss=losses) )
    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':

    # Parameters
    keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
    batch_size = 2
    workers = 4
    print_freq =1

    label_map, rev_label_map, label_color_map = label_map_fn('/content/data/output.json')

    n_classes = len(label_map)  # number of different types of objects

    model, device = model_init('SSD')
    model = model.to(device)
    # Switch to eval mode
    model.eval()

    test_dataset = COCODataset('/content/data/images', '/content/data/output.json' , split = 'test' , keep_difficult=keep_difficult)                             
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

    evaluate(test_loader, model, device)

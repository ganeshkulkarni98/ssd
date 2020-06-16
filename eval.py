from utils import *
from datasets import COCODataset
from tqdm import tqdm
from pprint import PrettyPrinter
from model import SSD300, VGGBase, MultiBoxLoss
import torchvision
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()


def model_init(model_name):
  if model_name == 'SSD':
    '''
    As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
    There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16 
    '''
    weight_file_path = '/content/ssd/vgg16-397923af.pth'
    #weight_file_path = '/content/ssd/CP_epoch1.pth'

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
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    losses = AverageMeter()
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    coco = get_coco_api_from_dataset(test_loader.dataset)
    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
    with torch.no_grad():
        # Batches
        for i, (images, targets) in enumerate(tqdm(test_loader, desc='Evaluating')):
            # print(images, targets)
            # images = images.to(device)  # (N, 3, 300, 300)
            outputs = {}
            outputs_boxes = []
            outputs_labels =[]
            outputs_scores = []
            images = list(img.to(device) for img in images)
            images = torch.stack(images, dim=0)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # # Store this batch's results for mAP calculation
            boxes = [t['boxes'] for t in targets]
            labels = [t['labels'] for t in targets]
            difficulties = [t['difficulties'] for t in targets]
            
            # boxes = [b.to(device) for b in boxes]
            # labels = [l.to(device) for l in labels]
            # difficulties = [d.to(device) for d in difficulties]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score, max_overlap,
                                                                                       top_k)

            for i in range(len(det_boxes_batch)):
              outputs_boxes += [j for j in det_boxes_batch[i]]
              outputs_labels += [j for j in det_labels_batch[i]]
              outputs_scores += [j for j in det_scores_batch[i]]
            #print(outputs_boxes, len(outputs_boxes))
            outputs['boxes'] = torch.stack(outputs_boxes, dim=0)
            outputs['labels'] = torch.stack(outputs_labels, dim=0)
            outputs['scores'] = torch.stack(outputs_scores, dim=0)
            outputs = [outputs]
            #outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

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

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)   

if __name__ == '__main__':

    # Parameters
    keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
    batch_size = 2
    workers = 4
    print_freq =1

    min_score=0.01
    max_overlap=0.45
    top_k=200

    label_map, rev_label_map, label_color_map = label_map_fn('/content/data/output.json')

    n_classes = len(label_map)  # number of different types of objects

    model, device = model_init('SSD')
    model = model.to(device)
    # Switch to eval mode
    model.eval()

    # Load test image folder with corresponding coco json file to test_dataset
    test_dataset = COCODataset('/content/data/images', '/content/data/output.json' , split = 'test' , keep_difficult=keep_difficult)                             
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


    evaluate(test_loader, model, device)

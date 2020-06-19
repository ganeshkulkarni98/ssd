from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torchvision
from model import SSD300, VGGBase
import numpy as np

def model_init(model_name):
  if model_name == 'SSD':
    '''
    As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
    There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16 
    '''
    #weight_file_path = '/content/ssd/vgg16-397923af.pth'
    #weight_file_path = '/content/ssd/CP_epoch1.pth'
    weight_file_path = '/content/ssd/SSD300_pretrained_weight.pth'

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

def detect(image_folder, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param image_folder: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    # Transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


    images = list(sorted(os.listdir(image_folder)))

    annotated_images = []
    results = []

    with torch.no_grad():
      for original_image in images:
        img_result = []
        #annotated_image_path = '/content/results/' + original_image
        original_image = Image.open(os.path.join(image_folder, original_image)).convert("RGB")

        # Transform
        image = normalize(to_tensor(resize(original_image)))

        # Move to default device
        image = image.to(device)

        # Forward prop.
        predicted_locs, predicted_scores = model(image.unsqueeze(0))

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                                max_overlap=max_overlap, top_k=top_k)

        # Move detections to the CPU
        det_boxes = det_boxes[0].to('cpu')
        #print(det_scores)
        # Transform to original image dimensions
        original_dims = torch.FloatTensor(
            [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
        det_boxes = det_boxes * original_dims

        # Decode class integer labels
        det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

        # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
        if det_labels == ['background']:
            # Just return original image
            
          annotated_images.append(np.array(image.cpu()))
          results.append([])

        else:      
          # Annotate
          annotated_image = original_image
          draw = ImageDraw.Draw(annotated_image)
          font = ImageFont.load_default()

          # Suppress specific classes, if needed
          for i in range(det_boxes.size(0)):
              objects = {}
              objects['x_min'] = det_boxes[i][0]
              objects['y_min'] = det_boxes[i][1]
              objects['x_max'] = det_boxes[i][2]
              objects['y_max'] = det_boxes[i][3]
              objects['conf_level'] = det_scores[0][i]
              objects['label']= det_labels[i]
              if suppress is not None:
                  if det_labels[i] in suppress:
                      continue

              img_result.append(objects)

              # Boxes
              box_location = det_boxes[i].tolist()
              draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
              draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
                  det_labels[i]]) 

              # Text
              text_size = font.getsize(det_labels[i].upper())
              text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
              textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                  box_location[1]]
              draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
              draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                        font=font)
          del draw

          # annotated_image.save(annotated_image_path)
          annotated_images.append(np.array(annotated_image))
          results.append(img_result)

    return annotated_images, results


if __name__ == '__main__':

    # hyperparameters
    min_score=0.8
    max_overlap=0.45
    top_k=5

    model_name = 'SSD'

    test_json_file = '/content/test_coco_dataset.json'
    test_images_folder = '/content/test_images'

    label_map, rev_label_map, label_color_map = label_map_fn(train_json_file)

    n_classes = len(label_map)  # number of different types of objects

    model, device = model_init(model_name)
    model = model.to(device)
    # Switch to eval mode
    model.eval()

    annotated_images, results = detect(image_folder, min_score, max_overlap, top_k)
    print(annotated_images, results)

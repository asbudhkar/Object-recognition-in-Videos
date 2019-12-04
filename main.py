from __future__ import print_function
import os
import numpy as np
import torch
from PIL import Image
import sys
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import argparse
from torchsummary import summary

os.system("git clone https://github.com/pytorch/vision.git")
os.system("git clone https://github.com/abewley/sort.git")
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './vision/references/detection')
import transforms as T
from engine import train_one_epoch, evaluate
import utils
import cv2
import matplotlib.pyplot as plt
import psutil
import os
import sort 
from sort import *

# Get device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Set root path where data is present
root_path = ''

class objDataset(object):
    def __init__(self, root, transforms, classes=None):
        self.root = root
        self.transforms = transforms
        self.classes = classes

        # load all images and dicts
        all_imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        all_dicts = list(sorted(os.listdir(os.path.join(root, "dicts"))))
        
        self.imgs = []
        self.dicts = []
        idx = 0
        for sd in all_dicts:
          for si in all_imgs:
            if sd[:-4]==si[:-4]:
              self.imgs.append(si)
              self.dicts.append(sd)
              break


    def __getitem__(self, idx):
        
        # Get image and dict path
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        dict_path = os.path.join(self.root, "dicts", self.dicts[idx])
        

        # Open image 
        img = Image.open(img_path).convert("RGB")
      
        # Open corresponding dictionary 
        f = open(dict_path, 'r')
        lines = f.readlines()
        f.close()
        
        # Get bounding box coordinates 
        box = [int(s) for s in lines[-1].split()]

        # Get bounding box label  
        label = lines[len(lines)-2].replace('\n','')
        
        boxes = torch.as_tensor([[box[0], box[1], box[0]+box[2], box[1]+box[3]]], dtype=torch.float32)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        
        # Get a list of classes 

        # TODO read different classes from file instead of command line arg
        label1 = args.classes.split(",") 
        i=len(label1) 
        
        while (i!=0): 
                labels = (label1.index(label)+1)*torch.ones((1,), dtype=torch.int64)    
                i=i-1
       
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Return predicted boxes and labels for each object

def get_prediction(model, img_path, threshold, classes, device=device):
  
  label=classes.split(",")
  LABELS = []
  # First label is background
  LABELS.append("background")
  for i in label:
      LABELS.append(i)  
  
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)

  # Get prediction
  with torch.no_grad():
    pred = model([img.to(device)])
  
  pred_score = list(pred[0]['scores'].detach().cpu().numpy())
  
  # Get prediction with score above threshold
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold]
  
  if pred_t:
    pred_t=pred_t[-1]
  else:
    return [],[]
 
  print(pred[0]['labels'])

  # Return class and box coordinates
  pred_class = [LABELS[i] for i in list(pred[0]['labels'].cpu().numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
  if(pred_t is []):
    return [],[]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes,pred_class


# Perform instance segmentation
def instance_segmentation_api(model, img_path, classes, threshold=0.5, rect_th=3, text_size=2, text_th=2,  device=device):
  
  # Save image with label and bounding box for each detected object
  boxes,classes = get_prediction(model, img_path, threshold, classes, device)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  for i in range(len(boxes)):
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
    cv2.putText(img, classes[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
  plt.figure(figsize=(20,30))
  plt.imsave('res',img)
  plt.xticks([])
  plt.yticks([])


# Get prediction for ach frame of video
def get_prediction_frame(model, frame, classes, threshold, device=device):
  
  label=classes.split(",")
  LABELS = []
  LABELS.append("background")
  for i in label:
      print(i)
      LABELS.append(i)  
  transform = T.Compose([T.ToTensor()])
  frame = transform(frame)

  # Get prediction
  with torch.no_grad():
    pred = model([frame.to(device)])


  pred_score = list(pred[0]['scores'].detach().cpu().numpy())

  # Get prediction with score above threshold
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
  print(i for i in list(pred[0]['labels'].cpu().numpy()))

  # Return predicted class and bounding box for each object detected
  pred_class = [LABELS[i] for i in list(pred[0]['labels'].cpu().numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes,pred_class

# Required for SORT as SORT input is in YOLO output form
def to_yolo_form(pred, pred_t):
  scores = pred[0]['scores'][:pred_t+1].unsqueeze(1)
  return torch.cat([pred[0]['boxes'][:pred_t+1,:], scores, scores, pred[0]['labels'][:pred_t+1].float().unsqueeze(1)],1)


# Perform instance segmentation
def video_instance_segmentation_sort(model, video_path, classes, threshold=0.5, rect_th=2, text_size=1, text_th=1,  device=device):
  
  vid = cv2.VideoCapture(video_path)
  label=classes.split(",")
  LABELS = []
  LABELS.append("background")
  for i in label:
      LABELS.append(i)  

  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  ret,frame=vid.read()

  outvideo = cv2.VideoWriter(video_path.replace(".mp4", "-hand.mp4"),fourcc,20.0,(int(vid.get(3)),int(vid.get(4))))

  since = time.time()
  frame_count = 0

  # Create SORT instance
  mot_tracker = Sort()

  while(True):
    ret, frame = vid.read()
    print(ret)
    if not ret:
      break
    frame_count+=1
    transform = T.Compose([T.ToTensor()])

    # Get prediction using model
    with torch.no_grad():
      pred = model([transform(frame).to(device)])

    # Get prediction with score above threshold
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    above_t = [pred_score.index(x) for x in pred_score if x>threshold]
    print(len(above_t))
    if len(above_t) > 0:   
      pred_t = above_t[-1]
      det = to_yolo_form(pred, pred_t)

      # Get tracked object
      tracked_objects = mot_tracker.update(det.cpu())
      boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(tracked_objects[:,0:4].astype(np.float32))]
      
      # Save frame with label and bounding box for each detected object
      pred_cls = [LABELS[i] for i in list(det[:,6].int())]
      frame = np.asarray(frame)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      for i in range(len(boxes)):
        cv2.rectangle(frame, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(frame,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      #cv2.imshow('result.png',frame) 
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
      break
    #outvideo.release()
    time_elapsed = time.time() - since
     
    print('Processing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Frames per second: {:.4f}'.format(time_elapsed / frame_count))
  
  outvideo.release()
  vid.release() 
  
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Load a trained detector and apply it to an image || or || Train a detector')
   
    parser.add_argument('--epochs', type=int, default=1, help='Epochs for training')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--classes', help='string with class dictionary')
    args = parser.parse_args()

    # Get fasterRCNN object from torchvision models
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    from torchvision import transforms as T
    model.to(device)

    classes = args.classes

    # Create list of classes 
    labels = classes.split(",")      
    num_classes = len(labels)+1  # N class + background
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Set root - Folder in which images and dists are present  
    root_path = '.'
    
    
    dataset = objDataset(root_path, get_transform(train=True), labels)
    dataset_test = objDataset(root_path, get_transform(train=False),labels)
    
    indices = torch.randperm(len(dataset)).tolist()
    
    # Split in train and test 

    #TODO modify the split as per the number of examples    
    dataset = torch.utils.data.Subset(dataset, indices[:500])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[500:600])

   
    # Define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
  
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = args.epochs
    
    for epoch in range(num_epochs):
        
        # Train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        
        # Update the learning rate
        lr_scheduler.step()
        
    torch.save(model.state_dict(), 'fasterRNN-hand.pt')
    
    # For testing
 
    #TODO create another file
    model.load_state_dict(torch.load('fasterRNN-hand.pt'))
    
    # Test random image from dataset 
    imgs = list(sorted(os.listdir(os.path.join(root_path, "images"))))
    img_path = os.path.join(root_path, "images", imgs[7350])

    model.eval()
    instance_segmentation_api(model, img_path,  args.classes, threshold=0.4) 
   
    # Test video
    video_instance_segmentation_sort(model, 'cam1.mp4', classes, threshold=0.5)

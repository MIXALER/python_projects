#!/usr/bin/env python
# coding: utf-8

# # Project V. Fish Detection with Deep Learning
# 1. Split Train and Val dataset
# 2. Train a detection model based on YOLOv3-tiny
# 3. Evaluate your model
# 4. Use your model to detect fish from images in data/samples

# ## Setup
# Please install required packages and make sure the version are valid
#
# pip install -r requirements.txt

# In[ ]:


from __future__ import division
import os
import shutil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from cv_class.proj5.utils.logger import *
from cv_class.proj5.utils.utils import *
from cv_class.proj5.utils.datasets import *
from cv_class.proj5.utils.augmentations import *
from cv_class.proj5.utils.transforms import *
from cv_class.proj5.utils.parse_config import *
from cv_class.proj5.utils.test import evaluate
from cv_class.proj5.utils.loss import compute_loss
from cv_class.proj5.utils.models import *

from terminaltables import AsciiTable
from matplotlib.ticker import NullLocator

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import random

if __name__ == '__main__':

    # # Data Preprocess
    # You should code this part first

    # In[ ]:

    #####################################################################################################
    #                                            Your Code                                              #
    #####################################################################################################
    # You should generate valid Train dataset and Val dataset.
    # Use data in data/custom/images and data/custom/labels to generate the path file train.txt and
    # val.txt in data/custom/
    # a qualified val dataset is smaller than the train dataset and
    # most time there are no overlapped data between two sets.
    path = 'data/custom/images'
    dir = os.listdir(path)
    output_path = 'output'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    fopen = open('data/custom/train.txt', 'w')
    fopen.truncate()
    for d in dir:
        string = 'data/custom/images/' + d + '\n'
        fopen.write(string)
    fvalid_open = open('data/custom/valid.txt', 'w')
    fvalid_open.truncate()
    fopen.close()
    with open('data/custom/train.txt') as f:
        lines = random.sample(f.readlines(), 121)
    fopen = open('data/custom/train.txt', 'w')
    fopen.truncate()
    fvalid_open.truncate()

    for i in range(len(lines)):
        s = str(lines[i]).replace('{', '').replace('}', '').replace("'", '').replace(':', ',')
        if (i <= 10):
            fvalid_open.write(s)
        else:
            fopen.write(s)
    fopen.close()
    fvalid_open.close()

    #####################################################################################################
    #                                                End                                                #
    #####################################################################################################

    # Make some config...

    # In[ ]:

    opt = {
        "epochs": 50,
        "model_def": "config/yolov3-tiny.cfg",
        "data_config": "config/custom.data",
        "pretrained_weights": "",
        "n_cpu": 1,
        "img_size": 416,
        "multiscale_training": True,
        "detect_image_folder": "data/samples"
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    # Get data configuration
    data_config = parse_data_config(opt["data_config"])
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    print(train_path)
    print(valid_path)
    print(class_names)

    # use pytorch to generate our model and dataset

    # Initiate model
    model = Darknet(opt["model_def"]).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt["pretrained_weights"] != "":
        if opt["pretrained_weights"].endswith(".pth"):
            model.load_state_dict(torch.load(opt["pretrained_weights"]))
        else:
            model.load_darknet_weights(opt["pretrained_weights"])

    # Get dataloader
    dataset = ListDataset(train_path, multiscale=opt["multiscale_training"], img_size=opt["img_size"],
                          transform=AUGMENTATION_TRANSFORMS)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=model.hyperparams['batch'] // model.hyperparams['subdivisions'],
        shuffle=True,
        # num_workers=opt["n_cpu"],
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # # Train your model!
    # You are required to complete the DL project training steps (get data batch from dataloader, forward, compute the loss and backward)
    # see more details in following comments.

    # In[ ]:

    for epoch in range(opt["epochs"]):
        print("\n---- Training Model ----")
        model.train()
        #####################################################################################################
        #                                            Your Code                                              #
        #####################################################################################################
        # Your code need to execute forward and backward steps.
        # Use 'enumerate' to get a batch[_, images, targets]
        # some helpful function
        # - outputs = model.__call__(imgs)(use it by model(imgs))
        # - loss, _ = cumpte_loss(outputs, targets, model)
        # - loss.backward() (backward step)
        # - optimizer.step() (execute params updating)
        # - optimizer.zero_grad() (reset gradients)
        # if you want to see how loss changes in each mini-batch step:
        # -eg print(f'Epoch:{epoch+1}, Step{step+1}/{len(dataloader)}, loss:{loss.item()}')
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device))
            optimizer.zero_grad()
            outputs = model(imgs)
            loss, _ = compute_loss(outputs, targets, model)
            loss.backward()
            optimizer.step()

        #####################################################################################################
        #                                                End                                                #
        #####################################################################################################

    # # Evaluate and save current model

    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    metrics_output = evaluate(
        model,
        path=valid_path,
        iou_thres=0.5,
        conf_thres=0.1,
        nms_thres=0.5,
        img_size=opt["img_size"],
        batch_size=model.hyperparams['batch'] // model.hyperparams['subdivisions'],
    )

    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        evaluation_metrics = [
            ("validation/precision", precision.mean()),
            ("validation/recall", recall.mean()),
            ("validation/mAP", AP.mean()),
            ("validation/f1", f1.mean()),
        ]
        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            print(class_names, c)
            ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean()}")
    else:
        print("---- mAP not measured (no detections found by model)")
    torch.save(model.state_dict(), f"checkpoints/yolov3-tiny_ckpt_%d.pth" % epoch)

    # # Detect and visualize results

    # In[ ]:

    model.eval()  # Set in evaluation mode
    dataloader = DataLoader(
        ImageFolder(opt["detect_image_folder"], transform= \
            transforms.Compose([DEFAULT_TRANSFORMS, Resize(opt["img_size"])])),
        batch_size=1,
        shuffle=False,
    )
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    print("\nPerforming object detection:")
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 0.2, 0.7)
        imgs.extend(img_paths)
        img_detections.extend(detections)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = detections.cpu()
            detections = rescale_boxes(detections, opt["img_size"], img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (class_names[int(cls_pred)], cls_conf.item()))
                box_w = x2 - x1
                box_h = y2 - y1
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=class_names[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )
        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = os.path.basename(path).split(".")[0]
        output_path = os.path.join("output", f"{filename}.jpg")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()

    # In[ ]:

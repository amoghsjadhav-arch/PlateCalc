# PlateCalc

A computer vision system that estimates the calorie content of a meal 
from a single food image.

## What it does

Upload a photo of a plate of food. PlateCalc detects each food item, 
estimates its portion size, and calculates the total calorie content 
using nutritional data from the USDA FoodData Central database.

## How it works

The pipeline has 4 stages:

**1. Detection & Segmentation**
A YOLOv8n-seg model trained on FoodSeg103 detects food items in the 
image and generates a segmentation mask for each one — a pixel-level 
outline of where each food is.

**2. Portion Estimation**
Foods are split into two categories:
- Countable items (eggs, apples, bananas etc): connected component 
  analysis counts individual instances, then multiplies by average 
  unit weight
- Region-based items (rice, curry, noodles etc): mask pixel area is 
  used as a proxy for portion size, assuming a full plate ≈ 500g

**3. Nutrition Lookup**
Each detected food label is queried against the USDA FoodData Central 
API to retrieve calories per 100g.

**4. Calorie Calculation**
calories = (estimated_weight_g / 100) × calories_per_100g
Individual values are summed to get total meal calories.

## Dataset & Training

- **Dataset**: FoodSeg103 — 7,118 food images across 103 categories 
  with pixel-level segmentation masks
- **Model**: YOLOv8n-seg (nano), fine-tuned from pretrained COCO weights
- **Training**: 50 epochs, batch size 16, image size 640×640, 2× T4 GPUs
- **mAP50**: ~0.xx after 50 epochs (updated after training)

## Design Choices & Trade-offs

**Why YOLOv8?**
YOLOv8 is a one-stage detector that handles detection and segmentation 
in a single forward pass. It's fast, well-documented, and the 
ultralytics library makes training on custom datasets straightforward. 
The nano variant was chosen to keep training feasible within free GPU 
quota while still demonstrating the full pipeline.

**Why pixel area as a proxy for portion size?**
Accurate portion estimation from a 2D image is an unsolved problem — 
it would require depth information or a reference object. Pixel area 
is a simple, interpretable heuristic that works reasonably for 
overhead shots of plates. The assumption is that a full plate ≈ 500g, 
so a food region covering 20% of the image ≈ 100g.

**Why separate countable and region-based foods?**
Pixel area breaks down for countable items — 3 eggs in a pile has 
roughly the same pixel area as 1 egg spread out. Connected component 
analysis on the mask gives a more accurate count of discrete items.

**Dataset conversion**
FoodSeg103 provides semantic segmentation masks (PNG files where each 
pixel value = class ID). YOLOv8 requires instance segmentation format 
(polygon coordinates per object). A custom conversion script uses 
OpenCV's findContours on per-class binary masks to extract polygon 
outlines, then normalizes coordinates to 0-1 range.

## Limitations

- **Accuracy**: mAP50 of ~0.238 reflects limited training. More epochs 
  and data augmentation would improve detection quality
- **Portion estimation**: pixel area assumes an overhead view of a flat 
  plate. Stacked foods or angled shots will give inaccurate estimates
- **USDA API**: generic search sometimes returns processed variants 
  (e.g. dried egg powder instead of fresh egg), affecting calorie 
  accuracy
- **Dataset bias**: FoodSeg103 is predominantly Chinese/Asian cuisine. 
  Performance on Western foods may be lower
- **Overlapping foods**: where foods overlap, masks may merge, affecting 
  both classification and portion estimates
- Model weights are not included in this repo as they exceed GitHub's
  file size limit. Run the training notebook on Kaggle with GPU T4 x2
  to reproduce them.

## Setup & Running

This project runs in a Kaggle Notebook with GPU enabled.

1. Open the notebook on Kaggle
2. Ensure GPU T4 and Internet are enabled in Settings
3. The FoodSeg103 dataset is attached as an input
4. Run all cells in order

For inference on a new image, update the image path in the 
`estimate_calories()` call and run the inference cells.

## Tech Stack

- YOLOv8 (ultralytics)
- OpenCV
- USDA FoodData Central API
- Gradio
- Python 3.12

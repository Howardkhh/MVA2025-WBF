from argparse import ArgumentParser
from pathlib import Path
import json
import os

from ensemble_boxes import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# methods = [nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion]
# ious = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0]
methods = [weighted_boxes_fusion]
ious = [0.4]
use_nwd = True

def get_name(method, iou, nwd):
    return f"{method.__name__}_{iou}" + ("_nwd" if nwd else "")

def main(args):
    if "private_test" in str(args.coco_ann):
        split = "private_test"
    elif "val" in str(args.coco_ann):
        split = "val"
    elif "pub_test" in str(args.coco_ann):
        split = "pub_test"
    elif "train" in str(args.coco_ann):
        split = "train"
    else:
        print("Warning: Cannot determine dataset split from input COCO path!!!!")
        split = "val"

    print(f"Using Weights: {args.weights}")
        
    results = {get_name(method, iou, use_nwd): [] for method in methods for iou in ious}
    coco_ann = COCO(args.coco_ann)
    coco_preds = []
    for coco_pred in args.coco_pred:
        coco_preds.append(coco_ann.loadRes(coco_pred))

    for img_id in tqdm(coco_ann.imgs):
        img_info = coco_ann.imgs[img_id]
        h, w = img_info['height'], img_info['width']

        boxes_lists, scores_lists, labels_lists = [], [], []

        for coco_pred in coco_preds:
            ann_ids = coco_pred.getAnnIds(img_id)
            preds = coco_pred.loadAnns(ann_ids)
            
            boxes_list, scores_list, labels_list = [], [], []
            for pred in preds:
                if pred["score"] < 0.1: continue
                x1, y1, x2, y2 = pred["bbox"][0], pred["bbox"][1], pred["bbox"][0]+pred["bbox"][2], pred["bbox"][1]+pred["bbox"][3]
                if x1 > w or x2 > w or y1 > h or y2 > h: 
                    # print(pred["bbox"], [x1, y1, x2, y2])
                    x1 = min(x1, w)
                    x2 = min(x2, w)
                    y1 = min(y1, h)
                    y2 = min(y2, h)
                boxes_list.append([x1/w, y1/h, x2/w, y2/h])
                scores_list.append(pred["score"])
                labels_list.append(pred["category_id"])

            boxes_lists.append(boxes_list)
            scores_lists.append(scores_list)
            labels_lists.append(labels_list)

        for method in methods:
            for iou in ious:
                boxes, scores, labels = method(boxes_lists, scores_lists, labels_lists, iou_thr=iou, nwd=use_nwd, weights=args.weights)
                boxes[:, ::2], boxes[:, 1::2] = boxes[:, ::2] * w, boxes[:, 1::2] * h
                for i in range(len(boxes)):
                    bbox = [boxes[i][0], boxes[i][1], boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1]]
                    results[get_name(method, iou, use_nwd)].append({"image_id": img_info["id"], "category_id": int(labels[i]), "bbox": [float(b) for b in bbox], "score": float(scores[i])})
    
    # print("------------------ Without NMS ------------------")
    # coco_eval = COCOeval(coco_ann, coco_pred, "bbox")
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()

    os.makedirs(f"{args.output_dir}/{split}", exist_ok=True)
    for method in methods:
        for iou in ious:
            with open(f"{args.output_dir}/{split}/{get_name(method, iou, use_nwd)}.json", "w") as f:
                json.dump(results[get_name(method, iou, use_nwd)], f)
            
            coco_pred_nms = coco_ann.loadRes(f"{args.output_dir}/{split}/{get_name(method, iou, use_nwd)}.json")
            
            print(f"------------------ With {method.__name__.upper()}, IOU: {iou}, NWD: {use_nwd} ------------------")
            coco_eval = COCOeval(coco_ann, coco_pred_nms, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--coco_ann", type=Path, required=True)
    parser.add_argument("--coco_pred", nargs='+', type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--weights", nargs='+', type=float, default=[])
    args = parser.parse_args()
    if args.weights == []:
        args.weights = [1] * len(args.coco_pred)
    main(args)
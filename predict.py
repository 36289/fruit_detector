import os
import numpy as np
from inference_sdk import InferenceHTTPClient
from collections import defaultdict
import xml.etree.ElementTree as ET
# import the inference-sdk
from inference_sdk import InferenceHTTPClient


# 创建推断客户端
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="cZ56uAVZQPbfu2ea5ROm"
)

# 指定包含图片和对应 XML 文件的目录路径
directory_path = r".\test"  # 替换为你的目录路径

# 定义计算 IoU 的函数
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2
    
    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou

# 初始化字典，用于保存每个类别的真阳性和假阳性数量
class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0, 'count': 0})

# 遍历目录中的所有 JPG 文件
for filename in os.listdir(directory_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(directory_path, filename)
        xml_path = os.path.join(directory_path, filename[:-4] + ".xml")  # 获取对应的 XML 文件路径
        
        if not os.path.exists(xml_path):
            print(f"没有找到图像 {filename} 的真实标签 (XML 文件)")
            continue  # 跳过没有真实标签的图像
        
        # print(f"正在处理 {image_path}...")
        result = CLIENT.infer(image_path, model_id="fruit-up71e/1")
        
        # 解析 XML 文件，获取真实标签数据
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ground_truth = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bbox = [int(obj.find('bndbox').find(coord).text) for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
            ground_truth.append({'class': class_name, 'bbox': bbox})
        
        # # 打印真实标签
        # print(f"真实标签: {ground_truth}")
        
        # 假设推断结果的格式与真实标签类似
        predictions = result['predictions']
        
        # # 打印预测结果
        # print(f"预测结果: {predictions}")
        
        for gt in ground_truth:
            gt_bbox = gt['bbox']
            gt_class = gt['class']
            
            gt_detected = False  # 记录该真实标签是否被检测到
            
            for pred in predictions:
                if 'x' in pred and 'y' in pred and 'width' in pred and 'height' in pred:
                    # 转换预测框格式：中心坐标和宽高 -> 左上角和右下角坐标
                    x_center, y_center = pred['x'], pred['y']
                    width, height = pred['width'], pred['height']
                    pred_bbox = [
                        int(x_center - width / 2),  # xmin
                        int(y_center - height / 2), # ymin
                        int(x_center + width / 2),  # xmax
                        int(y_center + height / 2)  # ymax
                    ]
                    pred_class = pred['class']
                    iou = compute_iou(pred_bbox, gt_bbox)
                    
                    
                    if iou >= 0.15 and pred_class == gt_class:
                        class_metrics[pred_class]['tp'] += 1
                        class_metrics[pred_class]['iou_sum'] += iou
                        class_metrics[pred_class]['count'] += 1
                        gt_detected = True
                        break
            if not gt_detected:
                class_metrics[gt_class]['fn'] += 1
        
        # 统计假阳性
        for pred in predictions:
            if 'x' in pred and 'y' in pred and 'width' in pred and 'height' in pred:
                pred_bbox = [
                    int(pred['x'] - pred['width'] / 2),  # xmin
                    int(pred['y'] - pred['height'] / 2), # ymin
                    int(pred['x'] + pred['width'] / 2),  # xmax
                    int(pred['y'] + pred['height'] / 2)  # ymax
                ]
                pred_class = pred['class']
                
                pred_detected = False
                for gt in ground_truth:
                    gt_bbox = gt['bbox']
                    iou = compute_iou(pred_bbox, gt_bbox)
                    if iou >= 0.15 and pred_class == gt['class']:
                        pred_detected = True
                        break
                
                if not pred_detected:
                    class_metrics[pred_class]['fp'] += 1

precisions = []
recalls = []
ious = []
overall_tp = 0
overall_fp = 0
overall_fn = 0
overall_iou_sum = 0
overall_count = 0
class_mAPs = {}

for class_name, metrics in class_metrics.items():
    tp = metrics['tp']
    fp = metrics['fp']
    fn = metrics['fn']
    iou_sum = metrics['iou_sum']
    count = metrics['count']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    iou = iou_sum / count if count > 0 else 0
    
    precisions.append(precision)
    recalls.append(recall)
    ious.append(iou)
    
    overall_tp += tp
    overall_fp += fp
    overall_fn += fn
    overall_iou_sum += iou_sum
    overall_count += count
    
    print(f"类别: {class_name}")
    print(f"精度 (Precision): {precision:.2f}")
    print(f"召回率 (Recall): {recall:.2f}")
    print(f"{class_name}的IoU: {iou:.2f}")
    print()

    # 计算单个类别的 mAP
    class_mAP = precision * recall if (precision + recall) > 0 else 0
    class_mAPs[class_name] = class_mAP

# 输出每个类别的 mAP
for class_name, class_mAP in class_mAPs.items():
    print(f"{class_name}的mAP: {class_mAP:.2f}")

# 计算整体的精度、召回率和 IoU
overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
overall_iou = overall_iou_sum / overall_count if overall_count > 0 else 0

print(f"整体精度 (Overall Precision): {overall_precision:.2f}")
print(f"整体召回率 (Overall Recall): {overall_recall:.2f}")
print(f"整体平均 IoU: {overall_iou:.2f}")

# 计算平均精度均值 (mAP)
mAP = np.mean(list(class_mAPs.values()))
print(f"平均精度均值 (mAP): {mAP:.2f}")


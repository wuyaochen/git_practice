import os
import cv2
import albumentations as A
import xml.etree.ElementTree as ET
from albumentations.pytorch import ToTensorV2

# 設定檔案路徑
image_path = "aug_test/images_path/quokka.jpg"
xml_path = "aug_test/labels_path/quokka.xml"
output_image_dir = "augmented_images"
output_xml_dir = "augmented_labels"

# 建立資料夾
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_xml_dir, exist_ok=True)

# 自動偵測目前最大的 frame 編號
existing_frames = [
    f for f in os.listdir(output_image_dir)
    if f.startswith("frame") and f.endswith(".jpg")
]

if existing_frames:
    max_id = max([int(f[5:-4]) for f in existing_frames])  # "frame123.jpg" → 123
    start_index = max_id + 1
else:
    start_index = 1

# 讀取圖片
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image.shape[:2]

# 讀取 XML 標籤
tree = ET.parse(xml_path)
root = tree.getroot()

# 取得所有標籤與bbox
bboxes = []
labels = []

#防止框超出圖片範圍
bbox_params = A.BboxParams(
    format='pascal_voc',
    label_fields=['category_ids'],
    min_visibility=0.3
)


for obj in root.findall("object"):
    label = obj.find("name").text
    xml_box = obj.find("bndbox")
    xmin = int(xml_box.find("xmin").text)
    ymin = int(xml_box.find("ymin").text)
    xmax = int(xml_box.find("xmax").text)
    ymax = int(xml_box.find("ymax").text)

    bboxes.append([xmin, ymin, xmax, ymax])
    labels.append(label)

# 定義要套用的增強操作
augmentations = [
    A.Compose([A.HorizontalFlip(p=1)], bbox_params=bbox_params),
    A.Compose([A.VerticalFlip(p=1)], bbox_params=bbox_params),
    A.Compose([A.ShiftScaleRotate(p=1, shift_limit=0.0625, scale_limit=0.1, rotate_limit=15)], bbox_params=bbox_params),
    A.Compose([A.RandomBrightnessContrast(p=1)], bbox_params=bbox_params),
    A.Compose([A.Blur(blur_limit=3, p=1)], bbox_params=bbox_params),
    A.Compose([A.GaussNoise(var_limit=(10.0, 50.0), p=1)], bbox_params=bbox_params),
    A.Compose([A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1)], bbox_params=bbox_params),
    A.Compose([A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1)], bbox_params=bbox_params),
]

# 原圖也輸出
all_images = [image]
all_bboxes = [bboxes]
all_labels = [labels]

# 增強圖片與標籤
for aug in augmentations:
    transformed = aug(image=image, bboxes=bboxes, category_ids=labels)
    all_images.append(transformed['image'])
    all_bboxes.append(transformed['bboxes'])
    all_labels.append(transformed['category_ids'])



# 輸出編號
for i, (img, boxes, lbls) in enumerate(zip(all_images, all_bboxes, all_labels)):
    # 輸出編號
    if not boxes:
        continue  # 若增強後沒有留下 bbox，就跳過這張圖

    frame_id = start_index + i
    filename = f"frame{frame_id}.jpg"
    xmlname = f"frame{frame_id}.xml"

    # 儲存圖片
    output_img_path = os.path.join(output_image_dir, filename)
    cv2.imwrite(output_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # 建立新的 XML 標籤
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = output_image_dir
    ET.SubElement(annotation, "filename").text = filename
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(annotation, "segmented").text = "0"
    
    for box, label in zip(boxes, lbls):
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(box[0]))
        ET.SubElement(bndbox, "ymin").text = str(int(box[1]))
        ET.SubElement(bndbox, "xmax").text = str(int(box[2]))
        ET.SubElement(bndbox, "ymax").text = str(int(box[3]))

    # 儲存 XML
    tree = ET.ElementTree(annotation)
    output_xml_path = os.path.join(output_xml_dir, xmlname)
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)

    print(f"已儲存圖片: {output_img_path}，對應標籤: {output_xml_path}")
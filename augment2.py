import os
import cv2
import albumentations as A
import xml.etree.ElementTree as ET

# 設定資料夾路徑
image_dir = "cut_datasets0411_aug/JPEGImages"
xml_dir = "cut_datasets0411_aug/Annotations"
output_image_dir = "cut_datasets0411_aug/augmented_images"
output_xml_dir = "cut_datasets0411_aug/augmented_labels"

# 建立輸出資料夾
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_xml_dir, exist_ok=True)

# 偵測目前最大 frame 編號
existing_frames = [
    f for f in os.listdir(output_image_dir)
    if f.startswith("frame") and f.endswith(".jpg")
]
if existing_frames:
    max_id = max([int(f[5:-4]) for f in existing_frames])
    frame_index = max_id + 1
else:
    frame_index = 1

# 定義增強操作
bbox_params = A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.3)
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

# 遍歷 JPEGImages 內的所有圖片
for filename in os.listdir(image_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    basename = os.path.splitext(filename)[0]
    image_path = os.path.join(image_dir, filename)
    xml_path = os.path.join(xml_dir, f"{basename}.xml")

    if not os.path.exists(xml_path):
        print(f"找不到對應的 XML 檔案: {xml_path}，跳過")
        continue

    # 讀取圖片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # 讀取 XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bboxes = []
    labels = []

    for obj in root.findall("object"):
        label = obj.find("name").text
        xml_box = obj.find("bndbox")
        xmin = int(xml_box.find("xmin").text)
        ymin = int(xml_box.find("ymin").text)
        xmax = int(xml_box.find("xmax").text)
        ymax = int(xml_box.find("ymax").text)
        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    # 原圖也保存一次
    transform_results = [(image, bboxes, labels)]

    # 套用增強
    for aug in augmentations:
        transformed = aug(image=image, bboxes=bboxes, category_ids=labels)
        if transformed["bboxes"]:
            transform_results.append((transformed["image"], transformed["bboxes"], transformed["category_ids"]))

    # 儲存增強後的圖片與標籤
    for img, boxes, lbls in transform_results:
        if not boxes:
            continue

        filename = f"frame{frame_index}.jpg"
        xmlname = f"frame{frame_index}.xml"

        # 儲存圖片
        cv2.imwrite(os.path.join(output_image_dir, filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # 建立 XML
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

        tree = ET.ElementTree(annotation)
        tree.write(os.path.join(output_xml_dir, xmlname), encoding='utf-8', xml_declaration=True)
        print(f"已儲存圖片: frame{frame_index}.jpg，標籤: frame{frame_index}.xml")

        frame_index += 1

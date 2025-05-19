# -*- coding: utf-8 -*-

from xml.dom import minidom
import os
import glob

lut={}
lut["NG"] =0



def convert_coordinates(size, box):
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1]-box[0]
    h = box[3]-box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_xml2yolo(lut, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))
    if not xml_files:
        print(f"警告：輸入資料夾 {input_dir} 中沒有找到任何 XML 文件")
        return
        
    for fname in xml_files:
        xmldoc = minidom.parse(fname)
        fname_base = os.path.basename(fname)[:-4]  # 移除 .xml
        fname_out = os.path.join(output_dir, fname_base + '.txt')

        with open(fname_out, "w") as f:

            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)

            for item in itemlist:
                # get class label
                classid =  (item.getElementsByTagName('name')[0]).firstChild.data
                if classid in lut:
                    label_str = str(lut[classid])
                else:
                    label_str = "-1"
                    print(f"警告：標籤 '{classid}' 不在對應表中，檔案：{fname}")
                    continue  # 跳過無效標籤
                    
                # get bbox coordinates
                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert_coordinates((width,height), b)
                #print(bb)

                f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')

        print ("wrote %s" % fname_out)



def main():
    input_dir = r"C:\Users\User\wyc\cut_datasets0411_aug\xml_labels\val"  # 替換為你的 XML 文件資料夾路徑
    output_dir = r"C:\Users\User\wyc\cut_datasets0411_aug\labels\val"  # 替換為你的 txt 文件儲存路徑
    convert_xml2yolo(lut, input_dir, output_dir)
    print(f"轉換完成！所有 YOLO 格式文件已儲存至：{output_dir}")

if __name__ == '__main__':
    main()
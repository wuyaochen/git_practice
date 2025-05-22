論文資料增強+xml轉yolo
順序如下：

1.將拍好的照片用dataset.py，儲存在JPEGImage資料夾裡  
2.用cut3.py將照片切割只保留中心管子部分  
3.LabelImg標注圖片存在JPEGImage跟Annotation(XML檔)  
4.使用augment2.py讀取資料夾內的照片與標注(augment1僅只有讀取單張照片)  
5.使用xml2yolo.py將Annotation的xml(Faster R-CNN)資料轉成txt(YOLO)資料  
6.將JPEGImage資料夾的圖片用photo_random.py傳到YOLO用資料夾Datasets，裡面的images中的三個資料夾a.Train b.Test c.Val  
7.使用label_move.py讀取images中的各個資料夾，傳到labels三個同名的資料夾  
8.完成，可以分別拿去跑Faster R-CNN跟YOLO


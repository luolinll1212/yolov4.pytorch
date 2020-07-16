# -*- coding: utf-8 -*-  

class config:
    # 数据集参数
    train_list = "./data/2007_train.txt"
    test_list = "./data/2007_test.txt"
    num_workers = 4
    batch_size = 4
    manualseed = 0
    img_h = 608
    img_w = 608
    classes = 20
    anchors = [[[142, 110], [192, 243], [459, 401]],
               [[36, 75], [76, 55], [72, 146]],
               [[12, 16], [19, 36], [40, 28]]]
    mosaic = True
    voc_classes = ["aeroplane","bicycle","bird","boat","bottle",
                   "bus","car","cat","chair","cow",
                   "diningtable","dog","horse","motorbike","person",
                   "pottedplant","sheep","sofa","train","tvmonitor"]

    

    # 训练参数
    num_epochs = 200
    start_epoch = 1
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999
    output = "output"
    pretrained = ""
    interval = 100
    valinterval = 20

    # detect
    confidence = 0.5
    num_thres = 0.3
    eval_pt = ""
    font = "./data/simhei.ttf"

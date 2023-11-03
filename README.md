<div align="center">
  <h2>SAM-Seg-VIS</h2>
  <p>Segment Anthing 系列模型分割结果可视化工具</p>
  <a href="#">
    <img alt="Python 3.9" src="https://img.shields.io/badge/python-3.9-blue.svg" />
  </a>
  <a href="#">
    <img alt="PyQt5 5.15.10" src="https://img.shields.io/badge/pyqt5-5.15.10-blue.svg" />
  </a>
  <a href="#">
    <img alt="Status" src="https://img.shields.io/badge/Status-Updating-green" />
  </a>
</div>

![image](https://github.com/chenluda/SAM-Seg-VIS/assets/45784833/3d3a4b66-7ad2-437e-80e1-097c8f44d08b)


## References

<b>Segment Anything in Medical Images</b> <br/>
Jun Ma<sup>1</sup>, Yuting He<sup>2</sup>, Feifei Li<sup>3</sup>, Lin Han<sup>4</sup>, Chenyu You<sup>5</sup>, Bo Wang<sup>6</sup><br/>
<sup>1 </sup>Peter Munk Cardiac Centre, University Health Network; Department of Laboratory Medicine and Pathobiology, University of Toronto; Vector Institute, Toronto, Canada  <br/>
<sup>2 </sup>the Department of Computer Science, Johns Hopkins University, USA<br/>
<sup>3 </sup>the Department of Cell and Systems Biology, University of Toronto, Canada<br/>
<sup>4 </sup>Tandon School of Engineering, New York University, USA<br/>
<sup>5 </sup>the Department of Electrical Engineering, Yale University, USA<br/>
<sup>6 </sup>Peter Munk Cardiac Centre, University Health Network; Department of Laboratory Medicine and Pathobiology and Department of Computer Science, University of Toronto; Vector Institute, Toronto, Canada<br/>
[paper](https://arxiv.org/abs/2304.12306) | [code](https://github.com/bowang-lab/MedSAM)

<b>Segment Anything</b> <br/>
Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick<br/>
Meta AI Research, FAIR<br/>
[paper](https://ai.facebook.com/research/publications/segment-anything/) | [code](https://github.com/facebookresearch/segment-anything)

<b>SAM-Med2D</b> <br/>
Junlong Cheng<sup>1,2</sup>, Jin Ye<sup>2</sup>, Zhongying Deng<sup>2</sup>, Jianpin Chen<sup>2</sup>, Tianbin Li<sup>2</sup>, Haoyu Wang<sup>2</sup>, Yanzhou Su<sup>2</sup>, Ziyan Huang<sup>2</sup>, Jilong Chen<sup>1</sup>, Lei Jiang<sup>1</sup>, Hui Sun<sup>2</sup>, Junjun He<sup>2</sup>, Shaoting Zhang<sup>2</sup>, Min Zhu<sup>1</sup>, Yu Qiao<sup>2</sup><br/>
<sup>1 </sup>Sichuan University<br/>
<sup>2 </sup>Shanghai AI Laboratory<br/>
[paper](https://arxiv.org/abs/2308.16184) | [code](https://github.com/OpenGVLab/SAM-Med2D)


## 运行环境

```
PyQt5 5.15.10
python 3.9
```

1. 下载模型权重至 checkpoints 的对应文件夹中

+ `checkpoints/MedSAM/medsam_vit_b.pth`: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link
+ `checkpoints/MedSAM/sam_vit_b_01ec64.pth`: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
+ `checkpoints/MedSAM/sam-med2d_b.pth`: https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link

2. 运行代码
```
python app.py
```
> **Note**
> ```
> └─ Models（SAM 系列模型结构放置地）
> 
>    ├─ MedSAM（模型名称）
> 
>    │ Copy from [MedSAM](https://github.com/bowang-lab/MedSAM/tree/main/segment_anything)
>
>    ├─ SAM
> 
>    │ Copy from [SAM](https://github.com/facebookresearch/segment-anything/tree/main/segment_anything)
> 
>    ├─ SAMMed
> 
>    │ Copy from [SAMMed](https://github.com/OpenGVLab/SAM-Med2D/tree/main/segment_anything)
> 
> └─ checkpoints（SAM 系列模型权重放置地）
> 
>    ├─ MedSAM
> 
>    │ medsam_vit_b.pth
>
>    ├─ SAM
> 
>    │ sam_vit_b_01ec64.pth
>
>    ├─ SAMMed
> 
>    │ sam-med2d_b.pth
> 
> └─ results（分割结果放置地）
>
>    ├─ example_bbox_84.0_58.0_127.0_100.0（图像名称 + 提示类型 + 坐标）
> 
>    │ example.png（原始图像）
>
>    │ example_GT.png（金标准图像）
>
>    │ example_medsam_mask.png（MedSAM 分割结果）
>
>    │ example_sam_mask.png（SAM 分割结果）
>
>    │ example_sammed_mask.png（SAMMed 分割结果）
> 
> └─ data（数据存放地）
> 
>    ├ example.png（示例图像）
> 
>    │ example_GT.png（示例图像金标准）
> 
> └─ annotation_info_file.txt（提示点、边框坐标存放地）
> 
> └─ app.py（主程序）
>  ```

## 更新日志

* 2023-11-03：上传项目。

## 待办

- [x] 单点，单框提示分割
- [ ] 多点，多框，点框提示分割
- [ ] 文本提示分割
- [ ] 加入非交互型分割大模型

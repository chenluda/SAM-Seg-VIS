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


## 运行环境

```
PyQt5 5.15.10
python 3.9
```
运行代码
```
python app.py
```
> **Note**
> ```
> └─ Models（SAM 系列模型**结构**放置地）
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
> └─ checkpoints（SAM 系列模型**权重**放置地）
> 
>    ├─ MedSAM
> 
>    │ Download the [model checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link) and place it at here, as 'checkpoints/MedSAM/medsam_vit_b.pth'.
>
>    ├─ SAM
> 
>    │ Download the [model checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it at here, as 'checkpoints/MedSAM/sam_vit_b_01ec64.pth'.
>
>    ├─ SAMMed
> 
>    │ Download the [model checkpoint](https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link) and place it at here, as 'checkpoints/MedSAM/sam-med2d_b.pth'.
> 
> └─ results（分割结果放置地）
>
>    ├─ example_bbox_84.0_58.0_127.0_100.0（图像名称+提示类型+坐标）
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

# Face and Lip detection and tracking

## 0. Samples
### 1. Face and Lip tracking
<div>
  <img src="https://github.com/jungwook518/Face_Lip_detection_tracking/blob/master/samples/Face_ex1.gif" style="float: left;" width="100" height="100">
  <img src="https://github.com/jungwook518/Face_Lip_detection_tracking/blob/master/samples/Lip_ex1.gif" style="float: right;" width="100" height="100">
</div>

## 1. Face detection and tracking
```python Face_tracking.py --video_path /input/videos/path/folder/ --save_video_path /save/videos/path/folder/ --mode tracking --save_label_path /save/keypoint/labeling/json/folder/ --check_video True```

## 2. Lip detection and tracking
```python Lip_tracking.py --video_path /input/videos/path/folder/ --save_video_path /save/videos/path/folder/ --mode tracking --save_label_path /save/keypoint/labeling/json/folder/ --check_video True```

| Time(s) | Tracking | Detection |
|:---:|:---:|:---:|
| Face | 1.1s | 11.2s |
| Lip | 0.9s | 10.1s |


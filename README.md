# Face and Lip detection and tracking

## Samples
### 1. Face tracking
<img src="https://github.com/jungwook518/Face_Lip_detection_tracking/blob/master/samples/Face_ex1.gif" width="100" height="100">

### 2. Face tracking
<img src="https://github.com/jungwook518/Face_Lip_detection_tracking/blob/master/samples/Lip_ex1.gif" width="100" height="100">

### 3. Test
<div id="the whole thing" style="height:100%; width:100%" >
    <div id="leftThing" style="position: relative; width:25%; background-color:blue;">
        <img src="https://github.com/jungwook518/Face_Lip_detection_tracking/blob/master/samples/Lip_ex1.gif" width="100" height="100">
    </div>

    <div id="content" style="position: relative; width:50%; background-color:green;">
        <img src="https://github.com/jungwook518/Face_Lip_detection_tracking/blob/master/samples/Lip_ex1.gif" width="100" height="100">
    </div>

    <div id="rightThing" style="position: relative; width:25%; background-color:yellow;">
        <img src="https://github.com/jungwook518/Face_Lip_detection_tracking/blob/master/samples/Lip_ex1.gif" width="100" height="100">
    </div>
</div>

## 1. Face detection and tracking
```python Face_tracking.py --video_path /input/videos/path/folder/ --save_video_path /save/videos/path/folder/ --mode tracking --save_label_path /save/keypoint/labeling/json/folder/ --check_video True```

## 2. Lip detection and tracking
```python Lip_tracking.py --video_path /input/videos/path/folder/ --save_video_path /save/videos/path/folder/ --mode tracking --save_label_path /save/keypoint/labeling/json/folder/ --check_video True```


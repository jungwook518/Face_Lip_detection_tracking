# Face and Lip detection and tracking

## 0. Samples
### 1. Face and Lip tracking
<div>
  <img src="https://github.com/jungwook518/Face_Lip_detection_tracking/blob/master/samples/Face_ex1.gif" style="float: left;" width="100" height="100">
  <img src="https://github.com/jungwook518/Face_Lip_detection_tracking/blob/master/samples/Lip_ex1.gif" style="float: right;" width="100" height="100">
</div>

## 1. Face detection and tracking
```python Face_tracking.py --video_path /input/videos/path/folder/ --save_video_path /save/videos/path/folder/ --mode tracking --save_label_path /save/keypoint/labeling/json/folder/ --check_video True```
1. --mode : tracking과 detection 존재
2. --check_video : True로 하면 crop된 video 생성이 되기에 감지가 어떻게 되었는지 확인할 수 있음.
3. --save_label_path : origin 영상의 각 frame마다의 감지된 좌표 생성.

## 2. Lip detection and tracking
```python Lip_tracking.py --video_path /input/videos/path/folder/ --save_video_path /save/videos/path/folder/ --mode tracking --save_label_path /save/keypoint/labeling/json/folder/ --check_video True```
1. --mode : tracking과 detection 존재
2. --check_video : True로 하면 crop된 video 생성이 되기에 감지가 어떻게 되었는지 확인할 수 있음.
3. --save_label_path : origin 영상의 각 frame마다의 감지된 좌표 생성.

## 3. Calculation time
| Time(s) | Tracking | Detection |
|:---:|:---:|:---:|
| Face | 1.1s | 11.2s |
| Lip | 0.9s | 10.1s |
1. Tracking이 Detection보다 약 10배 빠르다.
2. 그러나 덜 정확할 수 있다.
3. 정면만 바라보고 있으면 Tracking이 훨씬 유리하고, 얼굴이 움직이면 Detection이 유리할 수 있다.

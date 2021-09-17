import json
import pdb



def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou



a='/home/nas/user/jungwook/Face_Lip_detection_tracking/labeling/Lip_AJOSCF030023_6.json'
b='/home/nas/user/jungwook/Face_Lip_detection_tracking/labeling/Lip_AJOSCF030023_6_median.json'


with open(a, 'r') as f:
    Face_mid = json.load(f)

with open(b, 'r') as f:
    Face_csrt = json.load(f)



box1 = []
box2 = []
iou_list = []
for i in range(len(Face_mid['Lip_bounding_box'])):
    box1.append(Face_mid['Lip_bounding_box']['frame_'+str(i)]['xtl'])
    box1.append(Face_mid['Lip_bounding_box']['frame_'+str(i)]['ytl'])
    box1.append(Face_mid['Lip_bounding_box']['frame_'+str(i)]['xbl'])
    box1.append(Face_mid['Lip_bounding_box']['frame_'+str(i)]['ybl'])

    box2.append(Face_csrt['Lip_bounding_box']['frame_'+str(i)]['xtl'])
    box2.append(Face_csrt['Lip_bounding_box']['frame_'+str(i)]['ytl'])
    box2.append(Face_csrt['Lip_bounding_box']['frame_'+str(i)]['xbl'])
    box2.append(Face_csrt['Lip_bounding_box']['frame_'+str(i)]['ybl'])

    # pdb.set_trace()

    iou = IoU(box1, box2)
    iou_list.append(iou)

    box1 = []
    box2 = []

pdb.set_trace()

print("hi")


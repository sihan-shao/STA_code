from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import numpy as np
from mmcv.image import imread, imwrite
import xlwt
import pandas as pd


def show_result(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img_name = img
    img = mmcv.imread(img_name)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # print(labels)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    if not (show or out_file):
        return img
    save_det_bbox2excel(img_name,
                        bboxes,
                        labels,
                        class_names=class_names,
                        score_thr=score_thr)


def save_det_bbox2excel(img,
                        bboxes,
                        labels,
                        class_names=None,
                        score_thr=0):
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        # bbox_int : [Xmin, Ymin, Xmax, Ymax]
        Xmin = bbox_int[0]
        Ymin = bbox_int[1]
        Xmax = bbox_int[2]
        Ymax = bbox_int[3]
        label = class_names[label]
        img_name = img.split('/')[1]
        list_a.append([img_name, label, Xmin, Ymin, Xmax, Ymax])


config_file = '../defect_config/team1/cascade_rcnn_r101_fpn_1x_eca_moreanchor.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../defect_work_dirs/cascade_rcnn_dcn_r101_fpn_1x_eca(more anchor+noise)/epoch_22.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# inference with multiple images
img_dir = 'Images/'
save_dir = 'team1_results_img/'
im_names = os.listdir(img_dir)
# solve the problem of sort
im_names.sort(key=lambda x: int(x.split(".")[0]))
# im_names.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
# im_names.sort(key=lambda x: x[0])
# -------------------------------
list_a = []

for im_name in im_names:
    print('--------------------------------------------')
    print('--process {} in a directory--'.format(im_name))

    result = inference_detector(model, img_dir + im_name)

    show_result(img_dir + im_name, result, model.CLASSES, score_thr=0.8, out_file=save_dir + im_name)
    # print(imfo)


df_data = pd.DataFrame(list_a, columns=['图片名', '瑕疵类型',
                                     'xmin', 'ymin', 'xmax', 'ymax'])

df_data.to_excel('team1/detection_result.xlsx',
                 sheet_name='Sheet1',
                 index=False, header=True)



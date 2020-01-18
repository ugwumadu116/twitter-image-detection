from PIL import Image
from matplotlib import pyplot as plt
from io import StringIO
from collections import defaultdict
import zipfile
import tensorflow as tf
import tarfile
import sys
import six.moves.urllib as urllib
import os
import numpy as np
from utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import glob
sys.path.append("..")


pb_fname = "../models/research/fine_tuned_model/frozen_inference_graph.pb"
label_map_pbtxt_fname = '../object_detection_twitter/data/annotations/label_map.pbtxt'
PATH_TO_CKPT = pb_fname
PATH_TO_LABELS = label_map_pbtxt_fname


def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())


num_classes = get_num_classes(label_map_pbtxt_fname)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    try:
        return {"status":True,"data":np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)}
    except ValueError:
        return {"status":False, "dimension":f"{im_width} * {im_height}" }



def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def image_detection(image_path):
    image_paths=glob.glob(os.path.join(image_path, "*.*"))
    image_nps ={}
    for paths in image_paths:
        image = Image.open(paths)
        image_np = load_image_into_numpy_array(image)
        
        if image_np["status"] is False:
            image_nps = image_np
            dimension = image_nps["dimension"]
            return {"message":f"The dimension {dimension} of your image is off"}
        image_np_expanded = np.expand_dims(image_np["data"], axis=0)
        output_dict = run_inference_for_single_image(image_np["data"], detection_graph)
    boxes = output_dict['detection_boxes']
    max_boxes_to_draw = boxes.shape[0]
    scores = output_dict['detection_scores']
    min_score_thresh = .5
    detections = []
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            class_name = category_index[output_dict['detection_classes'][i]]['name']
            detections.append({"name":"Array of bounding boxes",
                              "cordinates":boxes[i].tolist(), "detection_class":(output_dict['detection_classes'][i].tolist())})
    return detections

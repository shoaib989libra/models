import os
import sys
import cv2
import tarfile
import zipfile
import argparse
import numpy as np
import tensorflow as tf
import six.moves.urllib as urllib

from PIL import Image
from io import StringIO
from collections import defaultdict
from matplotlib import pyplot as plt
from distutils.version import StrictVersion
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
from playsound import playsound



def predict_video(video, path):
	if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
		raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

	MODEL_NAME = 'inference_graph'
	PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
	PATH_TO_LABELS = 'training_faster_rcnn_inception_v2_pets/labelmap.pbtxt'

	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



	def run_inference_for_single_image(image, graph):
		if 'detection_masks' in tensor_dict:
			# The following processing is only for single image
			detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
			detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
			# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
			real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
			detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
			detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
			detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
				detection_masks, detection_boxes, image.shape[0], image.shape[1])
			detection_masks_reframed = tf.cast(
				tf.greater(detection_masks_reframed, 0.5), tf.uint8)
			# Follow the convention by adding back the batch dimension
			tensor_dict['detection_masks'] = tf.expand_dims(
				detection_masks_reframed, 0)
		image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

		# Run inference
		output_dict = sess.run(tensor_dict,
								feed_dict={image_tensor: np.expand_dims(image, 0)})

		# all outputs are float32 numpy arrays, so convert types as appropriate
		output_dict['num_detections'] = int(output_dict['num_detections'][0])
		output_dict['detection_classes'] = output_dict[
			'detection_classes'][0].astype(np.uint8)
		output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
		output_dict['detection_scores'] = output_dict['detection_scores'][0]
		if 'detection_masks' in output_dict:
			output_dict['detection_masks'] = output_dict['detection_masks'][0]
		return output_dict

	try:
		with detection_graph.as_default():
			with tf.Session() as sess:
				# Get handles to input and output tensors
				ops = tf.get_default_graph().get_operations()
				all_tensor_names = {output.name for op in ops for output in op.outputs}
				tensor_dict = {}
				for key in [
					'num_detections', 'detection_boxes', 'detection_scores',
					'detection_classes', 'detection_masks'
				]:
					tensor_name = key + ':0'
					if tensor_name in all_tensor_names:
						tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
						tensor_name)
				
				cap = cv2.VideoCapture(video)

				# Check if camera opened successfully
				if (cap.isOpened() == False): 
					print("Unable to read camera feed")
				
				# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
				width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
				height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
				fourcc = cv2.VideoWriter_fourcc(*'XVID')
				out = cv2.VideoWriter(path, fourcc, 30.0, (width, height))

				while cap.isOpened():
					ret, image_np = cap.read()
					
					image_np_expanded = np.expand_dims(image_np, axis=0)
					# Actual detection.
					output_dict = run_inference_for_single_image(image_np, detection_graph)
					# Visualization of the results of a detection.
					vis_util.visualize_boxes_and_labels_on_image_array(
						image_np,
						output_dict['detection_boxes'],
						output_dict['detection_classes'],
						output_dict['detection_scores'],
						category_index,
						instance_masks=output_dict.get('detection_masks'),
						use_normalized_coordinates=True,
						line_thickness=8,
						min_score_tresh=args.tresh)

					out.write(cv2.resize(image_np, (width, height)))
				
				cap.release()
				out.release()
				cv2.destroyAllWindows()
	
	except Exception as e:
		print(e)



def predict_livefeed():
	if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
		raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

	MODEL_NAME = 'inference_graph'
	PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
	PATH_TO_LABELS = 'training_faster_rcnn_inception_v2_pets/labelmap.pbtxt'

	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



	def run_inference_for_single_image(image, graph):
		if 'detection_masks' in tensor_dict:
			# The following processing is only for single image
			detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
			detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
			# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
			real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
			detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
			detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
			detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
				detection_masks, detection_boxes, image.shape[0], image.shape[1])
			detection_masks_reframed = tf.cast(
				tf.greater(detection_masks_reframed, 0.5), tf.uint8)
			# Follow the convention by adding back the batch dimension
			tensor_dict['detection_masks'] = tf.expand_dims(
				detection_masks_reframed, 0)
		image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

		# Run inference
		output_dict = sess.run(tensor_dict,
								feed_dict={image_tensor: np.expand_dims(image, 0)})

		# all outputs are float32 numpy arrays, so convert types as appropriate
		output_dict['num_detections'] = int(output_dict['num_detections'][0])
		output_dict['detection_classes'] = output_dict[
			'detection_classes'][0].astype(np.uint8)
		output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
		output_dict['detection_scores'] = output_dict['detection_scores'][0]
		if 'detection_masks' in output_dict:
			output_dict['detection_masks'] = output_dict['detection_masks'][0]
		return output_dict

	try:
		with detection_graph.as_default():
			with tf.Session() as sess:
				# Get handles to input and output tensors
				ops = tf.get_default_graph().get_operations()
				all_tensor_names = {output.name for op in ops for output in op.outputs}
				tensor_dict = {}
				for key in [
					'num_detections', 'detection_boxes', 'detection_scores',
					'detection_classes', 'detection_masks'
				]:
					tensor_name = key + ':0'
					if tensor_name in all_tensor_names:
						tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
						tensor_name)
				
				cap = cv2.VideoCapture(0)

				# Check if camera opened successfully
				if (cap.isOpened() == False): 
					print("Unable to read camera feed")
				
				# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
				width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
				height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

				while True:
					ret, image_np = cap.read()
					
					image_np_expanded = np.expand_dims(image_np, axis=0)
					# Actual detection.
					output_dict = run_inference_for_single_image(image_np, detection_graph)
					# Visualization of the results of a detection.
					vis_util.visualize_boxes_and_labels_on_image_array(
						image_np,
						output_dict['detection_boxes'],
						output_dict['detection_classes'],
						output_dict['detection_scores'],
						category_index,
						instance_masks=output_dict.get('detection_masks'),
						use_normalized_coordinates=True,
						line_thickness=8,
						min_score_tresh=args.tresh)

					cv2.imshow('Prediction', cv2.resize(image_np, (width, height)))
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
				
				cap.release()
				cv2.destroyAllWindows()
	
	except Exception as e:
		print(e)



def predict_images(img_dir, path):
	if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
		raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

	MODEL_NAME = 'inference_graph'
	PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
	PATH_TO_LABELS = 'training_faster_rcnn_inception_v2_pets/labelmap.pbtxt'

	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



	def run_inference_for_single_image(image, graph):
		if 'detection_masks' in tensor_dict:
			# The following processing is only for single image
			detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
			detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
			# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
			real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
			detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
			detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
			detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
				detection_masks, detection_boxes, image.shape[0], image.shape[1])
			detection_masks_reframed = tf.cast(
				tf.greater(detection_masks_reframed, 0.5), tf.uint8)
			# Follow the convention by adding back the batch dimension
			tensor_dict['detection_masks'] = tf.expand_dims(
				detection_masks_reframed, 0)
		image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

		# Run inference
		output_dict = sess.run(tensor_dict,
								feed_dict={image_tensor: np.expand_dims(image, 0)})

		# all outputs are float32 numpy arrays, so convert types as appropriate
		output_dict['num_detections'] = int(output_dict['num_detections'][0])
		output_dict['detection_classes'] = output_dict[
			'detection_classes'][0].astype(np.uint8)
		output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
		output_dict['detection_scores'] = output_dict['detection_scores'][0]
		if 'detection_masks' in output_dict:
			output_dict['detection_masks'] = output_dict['detection_masks'][0]
		return output_dict

	try:
		with detection_graph.as_default():
			with tf.Session() as sess:
				# Get handles to input and output tensors
				ops = tf.get_default_graph().get_operations()
				all_tensor_names = {output.name for op in ops for output in op.outputs}
				tensor_dict = {}
				for key in [
					'num_detections', 'detection_boxes', 'detection_scores',
					'detection_classes', 'detection_masks'
				]:
					tensor_name = key + ':0'
					if tensor_name in all_tensor_names:
						tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
						tensor_name)

				files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
				for i, file_name in enumerate(files):
					image_np = cv2.imread(os.path.join(img_dir, file_name))
					height, width, _ = image_np.shape
					
					image_np_expanded = np.expand_dims(image_np, axis=0)
					# Actual detection.
					output_dict = run_inference_for_single_image(image_np, detection_graph)
					# Visualization of the results of a detection.
					vis_util.visualize_boxes_and_labels_on_image_array(
						image_np,
						output_dict['detection_boxes'],
						output_dict['detection_classes'],
						output_dict['detection_scores'],
						category_index,
						instance_masks=output_dict.get('detection_masks'),
						use_normalized_coordinates=True,
						min_score_thresh=args.thresh)
					
					if len([True for score in output_dict['detection_scores'] if score > args.thresh]):
						print("Criminal, pistol detected, Alert guards!")
					else:
						print("Safe")
					
					cv2.imshow('Prediction', cv2.resize(image_np, (width, height)))
					playsound('siren.wav')
					cv2.waitKey(0)

					# cv2.imwrite(os.path.join(path, file_name), image_np)
					print(f'Processed {i + 1} out of {len(files)} images.')
	except Exception as e:
		print(e)



# Adding the keyword arguments
parser = argparse.ArgumentParser()
parser.add_argument('--type', '-t', help='Type of input to process', choices=['image', 'video', 'live'], required=True)
parser.add_argument('--input', '-i', help='Path to input file type')
parser.add_argument('--output', '-o', help='Path to output video')
parser.add_argument('--thresh', '-r', help='Threshold value', type=float, default=0.5)
args = parser.parse_args()

if __name__ == '__main__':
	# Error handling
	if args.input and not os.path.exists(args.input):
		sys.exit('Path given to input does not exist.')
	
	if args.type == 'image':
		if not os.path.exists(args.output):
			os.makedirs(args.output)
		predict_images(args.input, args.output)
	elif args.type == 'video':
		predict_video(args.input, args.output)
	else:
		predict_livefeed()
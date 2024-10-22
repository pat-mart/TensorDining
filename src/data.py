import os
import xml.etree.ElementTree as eT
import tensorflow as tf


def parse_xml(file, directory):
    tree = eT.parse(file)
    root = tree.getroot()

    img_file = root.find('filename').text
    image_path = os.path.join(directory, img_file)

    boxes = []
    labels = []

    for obj in root.findall('object'):
        label = obj.find('name').text

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Thank you PyCharm code completion!
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return image_path, boxes, labels

def load_preprocess_imgs(image_path, boxes, labels):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.convert_image_dtype(image, tf.float32)

    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.string)

    return image, boxes, labels

def get_dataset(files, img_dir):
    image_paths = []
    boxes_list = []
    labels_list = []

    for xml_file in files:
        image_path, boxes, labels = parse_xml(xml_file, img_dir)
        image_paths.append(image_path)
        boxes_list.append(boxes)
        labels_list.append(labels)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, boxes_list, labels_list))
    dataset = dataset.map(
        lambda img_path, bs, ls: load_preprocess_imgs(img_path, bs, ls),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.batch(8)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def get_xml_paths() -> [str]:
    files = os.listdir("../data/images/raw_pascalvoc")
    files = [f for f in files if f[-4:] == ".xml"]
    return files
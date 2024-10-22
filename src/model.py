import tensorflow_hub as hub
import keras

from src.data import get_dataset, get_xml_paths


def get_model():
    model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")

    input_image = keras.layers.Input(shape=(224, 224, 3))
    output = model(input_image)

    output_class = keras.layers.Dense(2, activation='softmax')(output['detection_classes'])

    # 4 outputs for each corner of bounding box
    output_boxes = keras.layers.Dense(4, activation='relu')(output['detection_boxes'])

    model = keras.Model(inputs=input_image, outputs=[output_class, output_boxes])
    return model

tensor_dining = get_model()
tensor_dining.compile(optimizer='adam', loss='categorical_crossentropy')

tensor_dining.fit(get_dataset(get_xml_paths(), "../data/images/raw"), epochs=2, steps_per_epoch=15, verbose=1)

# prediction code for my Streamlit app, kept in its own file so streamlit_app.py only does the page layout
import json
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
SETTINGS = json.loads((HERE / 'settings.json').read_text())
NAMES = json.loads((HERE / 'class_names.json').read_text())
SIZE = SETTINGS['size']
_compiled = {}


def load_model(path=None):
    # the model is too big for a normal GitHub file, so it's downloaded once from the release
    path = Path(path) if path else HERE / 'model.keras'
    if not path.exists():
        part = path.with_suffix('.part')
        urllib.request.urlretrieve(SETTINGS['model_url'], part)
        part.rename(path)
    return tf.keras.models.load_model(path, compile=False)


def preprocess(picture):
    # same as validation in my notebook: resize to size+32, then centre crop to size
    x = tf.image.resize(np.asarray(picture.convert('RGB')), (SIZE + 32, SIZE + 32), antialias=True)
    x = tf.clip_by_value(tf.round(x), 0, 255)
    x = tf.image.resize_with_crop_or_pad(x, SIZE, SIZE)
    return tf.cast(x, tf.float32)[None, ...]


def top5(model, picture):
    # one compiled graph per model, about 2 to 3 times faster on CPU than calling the model layer by layer
    if id(model) not in _compiled:
        _compiled[id(model)] = tf.function(lambda x: model(x, training=False))
    predict = _compiled[id(model)]
    x = preprocess(picture)
    probs = predict(x).numpy()[0]
    if SETTINGS['flip_tta']:
        # average with the mirrored photo, same as I picked on validation
        probs = 0.5 * (probs + predict(tf.image.flip_left_right(x)).numpy()[0])
    best = np.argsort(probs)[::-1][:5]
    return [(NAMES[i], float(probs[i])) for i in best]


def _gradcam_parts(model):
    # my model is: input -> backbone (ConvNeXtBase, ends with pooling + LayerNorm) -> dropout -> Dense(200)
    # for Grad-CAM I need the last 7x7 feature map, which only exists inside the backbone right before its pooling
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    pool_index = next(i for i, layer in enumerate(backbone.layers)
                      if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D))
    features = tf.keras.Model(backbone.input, backbone.layers[pool_index].input)
    # everything after the feature map: the backbone's own pooling and LayerNorm, then my dropout and Dense head
    after_pool = backbone.layers[pool_index:]
    head = model.layers[model.layers.index(backbone) + 1:]

    @tf.function
    def cam(x):
        with tf.GradientTape() as tape:
            maps = features(x, training=False)
            tape.watch(maps)
            y = maps
            for layer in after_pool + head:
                y = layer(y, training=False)
            top = tf.argmax(y[0])
            score = tf.math.log(y[0, top] + 1e-9)  # log of the top probability, gives steadier gradients
        # Grad-CAM: weight each of the 1024 feature channels by how much it raised the top class score,
        # add them up, and keep only the positive part (the areas that pushed towards this species)
        grads = tape.gradient(score, maps)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        heat = tf.nn.relu(tf.reduce_sum(weights * maps, axis=-1))[0]
        return heat / (tf.reduce_max(heat) + 1e-8)

    return cam


def gradcam(model, picture, strength=0.45):
    # returns the cropped photo the model saw with a heatmap on top: red = mattered most for the top guess
    if ('cam', id(model)) not in _compiled:
        _compiled[('cam', id(model))] = _gradcam_parts(model)
    x = preprocess(picture)
    heat = _compiled[('cam', id(model))](x)
    # the map is only 7x7 for a 224px photo, so it's smoothed up to full size
    heat = tf.image.resize(heat[..., None], (SIZE, SIZE), method='bicubic').numpy()[..., 0].clip(0, 1)
    colour = np.stack([np.clip(1.5 - abs(4 * heat - k), 0, 1) for k in (3, 2, 1)], axis=-1) * 255  # simple jet colours
    photo = x[0].numpy()
    return ((1 - strength) * photo + strength * colour).astype(np.uint8)

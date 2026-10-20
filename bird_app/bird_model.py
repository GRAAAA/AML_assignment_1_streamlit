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

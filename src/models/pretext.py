from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Input

from src.models.layers import LinearClassifier


def create_pretext_model(
    img_height: int,
    img_width: int,
    backbone: Model,
    nclasses: int = 4,
    classification: bool = True,
    ) -> Model:

    input_layer = Input(shape=(img_height, img_width, 1))

    flatten = Flatten()
    classifier = LinearClassifier(nclasses, classification=classification)

    out = backbone(input_layer)
    out = flatten(out)
    out = classifier(out)

    return Model(inputs=input_layer, outputs=out, name="PretextModel")

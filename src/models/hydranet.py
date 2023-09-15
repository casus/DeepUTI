from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Input, Softmax

from src.models.layers import PixelClassifier
from src.models.modules import create_decoder, create_encoder


def create_hydranet(
    img_height: int,
    img_width: int,
    filters: int = 8,
    do: float = 0.2,
    l2: float = 1e-4,
    num_classes: int = 8,
):
    input_layer = Input(shape=(img_height, img_width, 1))

    encoder = create_encoder(img_height, img_width, filters=filters, do=do, l2=l2)

    residual_layers = ["conv1", "conv2", "conv3", "conv4"]
    encoder_with_residuals = Model(
        inputs=encoder.input,
        outputs=[encoder.get_layer(layer_name).output for layer_name in residual_layers] + encoder.outputs
    )

    out = encoder_with_residuals(input_layer)

    outputs = []
    for idx in range(num_classes):
        decoder = create_decoder(img_height, img_width, filters=filters, do=do, l2=l2, name=f"decoder_{idx}")
        classifier = PixelClassifier(1)
        outputs.append(classifier(decoder(out)))

    concat = Concatenate(-1)
    out = concat(outputs)

    activation = Softmax(axis=-1)
    out = activation(out)

    return Model(inputs=input_layer, outputs=out, name="HydraNet")

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer

from src.models.modules import create_decoder, create_encoder


def create_unet_backbone(
    img_height: int,
    img_width: int,
    filters: int = 8,
    do: float = 0.2,
    l2: float = 1e-4,
    use_residuals: bool = True,
) -> dict[str, Model]:
    input_layer = Input(shape=(img_height, img_width, 1))

    encoder = create_encoder(img_height, img_width, filters=filters, do=do, l2=l2)

    if use_residuals:
        residual_layers = ["conv1", "conv2", "conv3", "conv4"]
        encoder = Model(
            inputs=encoder.input,
            outputs=[encoder.get_layer(layer_name).output for layer_name in residual_layers] + encoder.outputs
        )

    decoder = create_decoder(
        img_height, img_width, filters=filters, do=do, l2=l2, use_residuals=use_residuals
    )

    out = encoder(input_layer)
    out = decoder(out)

    backbone = Model(inputs=input_layer, outputs=out, name="Unet7")

    return {
        "backbone": backbone,
        "encoder": encoder,
        "decoder": decoder,
    }


def add_classifier(
    img_height: int,
    img_width: int,
    backbone: Model,
    classifier: Layer,
    ):
    input_layer = Input(shape=(img_height, img_width, 1))

    out = backbone(input_layer)
    out = classifier(out)

    return Model(inputs=input_layer, outputs=out, name="Unet7")

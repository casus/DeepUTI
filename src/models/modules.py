
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, MaxPooling2D, SpatialDropout2D
from tensorflow_addons.layers import InstanceNormalization

from src.models.layers import ConvBlock, DeconvBlock


def create_encoder(img_height, img_width, filters=8, do=0.2, l2=1e-4):
    input_layer = Input(shape=(img_height, img_width, 1), name="image_input")

    conv1 = ConvBlock(nfilters=filters, l2=l2, name="conv1")
    norm1 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )
    pool1 = MaxPooling2D(pool_size=(2, 2))
    do1 = SpatialDropout2D(do)

    conv2 = ConvBlock(nfilters=filters*2, l2=l2, name="conv2")
    norm2 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )
    pool2 = MaxPooling2D(pool_size=(2, 2))
    do2 = SpatialDropout2D(do)

    conv3 = ConvBlock(nfilters=filters*4, l2=l2, name="conv3")
    norm3 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )
    pool3 = MaxPooling2D(pool_size=(2, 2))
    do3 = SpatialDropout2D(do)

    conv4 = ConvBlock(nfilters=filters*8, l2=l2, name="conv4")
    norm4 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )
    pool4 = MaxPooling2D(pool_size=(2, 2))
    do4 = SpatialDropout2D(do)

    conv5 = ConvBlock(nfilters=filters*16, l2=l2, name="conv5")
    norm5 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )
    do5 = SpatialDropout2D(do)

    out = do1(pool1(norm1(conv1(input_layer))))
    out = do2(pool2(norm2(conv2(out))))
    out = do3(pool3(norm3(conv3(out))))
    out = do4(pool4(norm4(conv4(out))))
    out = do5(norm5(conv5(out)))

    model = Model(inputs=input_layer, outputs=out, name="Encoder")

    return model

def create_decoder(
    img_height,
    img_width,
    filters=8,
    do=0.2,
    l2=1e-4,
    name="decoder",
    use_residuals=True,
    ):

    deconv1 = DeconvBlock(nfilters=filters*8, l2=l2)
    norm1 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )
    do1 = SpatialDropout2D(do)

    deconv2 = DeconvBlock(nfilters=filters*4, l2=l2)
    norm2 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )
    do2 = SpatialDropout2D(do)

    deconv3 = DeconvBlock(nfilters=filters*2, l2=l2)
    norm3 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )
    do3 = SpatialDropout2D(do)

    deconv4 = DeconvBlock(nfilters=filters, l2=l2)
    norm4 = InstanceNormalization(
        axis=-1,
        center=True,
        scale=True,
        beta_initializer="random_uniform",
        gamma_initializer="random_uniform",
    )


    if use_residuals:
        inputs = []
        for power in range(5):
            mult = 2**power
            inputs.append(
                Input(shape=(img_height//mult, img_width//mult, filters*mult))
            )

        out = do1(norm1(deconv1(inputs[4], inputs[3])))
        out = do2(norm2(deconv2(out, inputs[2])))
        out = do3(norm3(deconv3(out, inputs[1])))
        out = norm4(deconv4(out, inputs[0]))
    else:
        mult = 2**4
        inputs = Input(shape=(img_height//mult, img_width//mult, filters*mult))

        out = do1(norm1(deconv1(inputs)))
        out = do2(norm2(deconv2(out)))
        out = do3(norm3(deconv3(out)))
        out = norm4(deconv4(out))


    model = Model(inputs=inputs, outputs=out, name=name)
    return model

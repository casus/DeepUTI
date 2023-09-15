import tensorflow as tf


def restore_checkpoint(weights_dir, **kwargs):
    checkpoint = tf.train.Checkpoint(**kwargs)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=weights_dir,
        max_to_keep=None,
    )
    manager.restore_or_initialize()

    return checkpoint

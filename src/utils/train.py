from tensorflow.keras.callbacks import LearningRateScheduler


def get_scheduler(start_after, step_size, gamma):
    def scheduler(epoch, lr):
        if (epoch > start_after and epoch % step_size == 0):
            return lr * gamma
        else:
            return lr

    return LearningRateScheduler(scheduler)

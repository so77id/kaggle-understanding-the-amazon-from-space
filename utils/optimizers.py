from keras.optimizers import RMSprop


def optimizer_factory(optimizer_params):
    if optimizer_params.name == "rmsprop":
        optimizer = RMSprop(lr=float(optimizer_params.lr), rho=float(optimizer_params.rho), epsilon=float(optimizer_params.epsilon), decay=float(optimizer_params.decay))

    return optimizer
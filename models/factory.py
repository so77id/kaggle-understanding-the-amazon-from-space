from models.inception_v4 import inception_v4_model


def model_factory(model_name, img_rows, img_cols, channel, num_classes, dropout_keep_prob, checkpoint=""):

    if model_name == 'inception_v4':
        model = inception_v4_model(img_rows, img_cols, channel, num_classes, dropout_keep_prob=dropout_keep_prob)

    if checkpoint != '':
        model.load_weights(checkpoint, by_name=True)

    return model
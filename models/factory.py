from models.inception_v4 import inception_v4_model
from models.resnet_50 import resnet50_model
from models.densenet_121 import densenet121_model


def model_factory(model_name, img_rows, img_cols, channel, num_classes, dropout_keep_prob=0, checkpoint=""):

    if model_name == 'inception_v4':
        model = inception_v4_model(img_rows, img_cols, channel, num_classes, dropout_keep_prob=dropout_keep_prob)
    elif model_name == 'resnet_50':
        model = resnet50_model(img_rows, img_cols, channel, num_classes)
    elif model_name == 'densenet_121':
        model = densenet121_model(img_rows, img_cols, channel, num_classes=num_classes, dropout_rate=dropout_keep_prob)

    if checkpoint != '':
        print("Loading checkpoint:", checkpoint)
        model.load_weights(checkpoint, by_name=True)

    return model
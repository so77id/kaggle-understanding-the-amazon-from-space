
import json

from collections import namedtuple


def load_configuration(configuration_file):
    with open(configuration_file, 'r') as content_file:
        content = content_file.read()

    return json.loads(content, object_hook=lambda d: namedtuple('Configuration', d.keys())(*d.values()))
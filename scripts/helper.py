from tensorflow.keras.utils import to_categorical

def preprocess_data(x_train, labels_train, x_test, labels_test):
    """Preprocesses the input data for a machine learning model.

    Args:
        x_train (numpy.ndarray): Training data features.
        labels_train (numpy.ndarray): Labels corresponding to training data.
        x_test (numpy.ndarray): Testing data features.
        labels_test (numpy.ndarray): Labels corresponding to testing data.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: 
        A tuple containing preprocessed training features, one-hot encoded training labels,
        preprocessed testing features, and one-hot encoded testing labels.
    """
    x_train = _normalize_data(x_train)
    x_test = _normalize_data(x_test)

    y_train = to_categorical(labels_train, 10)
    y_test = to_categorical(labels_test, 10)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    return (x_train, y_train, x_test, y_test)


def _normalize_data(x):
    x = x.astype('float32')
    return x / 255
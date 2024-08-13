import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, InputLayer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features = np.load(os.path.join(script_dir, 'final_features.npy'), allow_pickle=True).astype(np.float32)
    labels = np.load(os.path.join(script_dir, 'final_labels.npy'), allow_pickle=True).astype(np.float32)
    return features, labels


def prepare_data(features, labels):
    num_classes = len(np.unique(labels))
    labels = labels.astype(int)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)

    features = np.expand_dims(features, axis=-1)

    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def main():
    features, labels = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = prepare_data(features, labels)

    input_shape = X_train.shape[1:] 

    model = build_model(input_shape, num_classes)
    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model.save(os.path.join(script_dir, 'audio_language_classifier.h5'))


if __name__ == "__main__":
    main()

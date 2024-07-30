import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, InputLayer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Load the feature and label data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features = np.load(os.path.join(script_dir, 'final_features.npy'), allow_pickle=True).astype(np.float32)
    labels = np.load(os.path.join(script_dir, 'final_labels.npy'), allow_pickle=True).astype(np.float32)
    return features, labels


# Prepare the data
def prepare_data(features, labels):
    # Ensure the labels are in integer format for one-hot encoding
    num_classes = len(np.unique(labels))
    labels = labels.astype(int)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)

    # Add a channel dimension to the features
    features = np.expand_dims(features, axis=-1)  # This adds the channel dimension

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


# Build the model
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))  # Update this line if needed

    # Convolutional layers
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

    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Main function to load data, prepare it, build the model, and train
def main():
    features, labels = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = prepare_data(features, labels)

    # Define the input shape based on the feature shape
    input_shape = X_train.shape[1:]  # (20, 500, 1) after adding channel dimension

    # Build and summarize the model
    model = build_model(input_shape, num_classes)
    model.summary()

    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',  # You can also monitor 'val_accuracy'
        patience=5,  # Number of epochs with no improvement to wait
        restore_best_weights=True
    )

    # Train the model with early stopping
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')

    # Save the model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model.save(os.path.join(script_dir, 'audio_language_classifier.h5'))


if __name__ == "__main__":
    main()

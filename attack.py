import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Layer
from sklearn.preprocessing import StandardScaler

# Custom Layers
class DynamicThresholdLayer(Layer):
    def __init__(self, threshold_initializer='zeros', **kwargs):
        super(DynamicThresholdLayer, self).__init__(**kwargs)
        self.threshold_initializer = tf.keras.initializers.get(threshold_initializer)
        self.threshold = None

    def build(self, input_shape):
        self.threshold = self.add_weight(name='threshold',
                                         shape=(),
                                         initializer=self.threshold_initializer,
                                         trainable=True)

    def call(self, inputs):
        return tf.cast(inputs > self.threshold, dtype=tf.float32)

    def get_config(self):
        config = super(DynamicThresholdLayer, self).get_config()
        config.update({'threshold_initializer': self.threshold_initializer})
        return config

class AttentionLayer(Layer):
    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.attention_weights = None

    def build(self, input_shape):
        self.attention_weights = self.add_weight(name='attention_weights',
                                                 shape=(input_shape[-1], self.units),
                                                 initializer='random_normal',
                                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.attention_weights)

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config

class FeatureSelectionLayer(Layer):
    def __init__(self, units, **kwargs):
        super(FeatureSelectionLayer, self).__init__(**kwargs)
        self.units = units
        self.feature_weights = None

    def build(self, input_shape):
        self.feature_weights = self.add_weight(name='feature_weights',
                                               shape=(input_shape[-1], self.units),
                                               initializer='random_normal',
                                               trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.feature_weights)

    def get_config(self):
        config = super(FeatureSelectionLayer, self).get_config()
        config.update({'units': self.units})
        return config

# Data Loading and Preprocessing
def load_and_preprocess_data():
    # Load datasets
    df1 = pd.read_csv(r'C:\Users\Anirudh\Desktop\Exeter\MSc Project\Dataset\DDoS\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', low_memory=False)
    df2 = pd.read_csv(r'C:\Users\Anirudh\Desktop\Exeter\MSc Project\Dataset\DDoS\Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv', low_memory=False)
    df3 = pd.read_csv(r'C:\Users\Anirudh\Desktop\Exeter\MSc Project\Dataset\DDoS\Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv', low_memory=False)

    # Combine datasets
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    shuffled_df = combined_df.sample(frac=1, random_state=42)

    # Encode labels
    label_encoder = LabelEncoder()
    shuffled_df['Label'] = label_encoder.fit_transform(shuffled_df['Label'])
    class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Class mapping: ", class_mapping)

    # Balance dataset
    balanced_df = resample(shuffled_df, n_samples=20000, random_state=123, replace=True)
    train_df, test_df = train_test_split(balanced_df, test_size=0.1, random_state=123)

    # Feature and label preparation
    columns_to_drop = ["Timestamp", "Protocol", "PSH Flag Cnt", "Init Fwd Win Byts", "Flow Byts/s", "Flow Pkts/s", "Label"]
    X_train = train_df.drop(columns=columns_to_drop).values.astype('float32')
    X_test = test_df.drop(columns=columns_to_drop).values.astype('float32')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_classes = len(np.unique(shuffled_df['Label']))
    y_train = to_categorical(train_df['Label'], num_classes=num_classes)
    y_test = to_categorical(test_df['Label'], num_classes=num_classes)

    return X_train_scaled, X_test_scaled, y_train, y_test, num_classes

# Custom Loss Function
def custom_loss(y_true, y_pred):
    cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    max_prediction = tf.reduce_max(y_pred, axis=-1)
    certainty_penalty = tf.reduce_mean(1 - max_prediction)
    total_loss = cross_entropy_loss + 0.1 * certainty_penalty
    return total_loss

# Building Autoencoders
def build_autoencoder(input_shape, encoding_dims):
    inputs = Input(shape=input_shape)
    encoded = inputs
    for dim in encoding_dims:
        encoded = Dense(dim, activation='relu')(encoded)
    decoded = encoded
    for dim in reversed(encoding_dims[:-1]):
        decoded = Dense(dim, activation='relu')(decoded)
    decoded = Dense(input_shape[0], activation='sigmoid')(decoded)
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    return autoencoder, encoder

# Building Final ANN Model
def build_final_ann(input_shapes, num_classes):
    inputs = [Input(shape=(shape,)) for shape in input_shapes]
    encoded_outputs = [Dense(64, activation='relu')(input) for input in inputs]
    concatenated = Concatenate()(encoded_outputs)
    x = Dense(64, activation='relu')(concatenated)
    x = Dense(32, activation='relu')(x)
    x = FeatureSelectionLayer(units=32)(x)
    x = AttentionLayer(units=16)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# Data loading
X_train_scaled, X_test_scaled, y_train, y_test, num_classes = load_and_preprocess_data()

# Building and training hierarchical autoencoders
shallow_autoencoder, shallow_encoder = build_autoencoder((X_train_scaled.shape[1],), [128, 64])
intermediate_autoencoder, intermediate_encoder = build_autoencoder((64,), [64, 32])
deep_autoencoder, deep_encoder = build_autoencoder((32,), [32, 16])

shallow_autoencoder.compile(optimizer='adam', loss='mse')
intermediate_autoencoder.compile(optimizer='adam', loss='mse')
deep_autoencoder.compile(optimizer='adam', loss='mse')

history_shallow = shallow_autoencoder.fit(X_train_scaled, X_train_scaled, epochs=10, batch_size=32, validation_data=(X_test_scaled, X_test_scaled))
encoded_train_shallow = shallow_encoder.predict(X_train_scaled)
encoded_test_shallow = shallow_encoder.predict(X_test_scaled)

history_intermediate = intermediate_autoencoder.fit(encoded_train_shallow, encoded_train_shallow, epochs=10, batch_size=32, validation_data=(encoded_test_shallow, encoded_test_shallow))
encoded_train_intermediate = intermediate_encoder.predict(encoded_train_shallow)
encoded_test_intermediate = intermediate_encoder.predict(encoded_test_shallow)

history_deep = deep_autoencoder.fit(encoded_train_intermediate, encoded_train_intermediate, epochs=10, batch_size=32, validation_data=(encoded_test_intermediate, encoded_test_intermediate))
encoded_train_deep = deep_encoder.predict(encoded_train_intermediate)
encoded_test_deep = deep_encoder.predict(encoded_test_intermediate)

# Building and training final ANN model
final_ann_model = build_final_ann([64, 32, 16], num_classes)
final_ann_model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
history_final = final_ann_model.fit([encoded_train_shallow, encoded_train_intermediate, encoded_train_deep],
                                    y_train,
                                    epochs=10,
                                    batch_size=32,
                                    validation_data=([encoded_test_shallow, encoded_test_intermediate, encoded_test_deep], y_test))

# Evaluation
scores = final_ann_model.evaluate([encoded_test_shallow, encoded_test_intermediate, encoded_test_deep], y_test)
print(f'Test accuracy: {scores[1]}')

# Predictions
predictions = final_ann_model.predict([encoded_test_shallow, encoded_test_intermediate, encoded_test_deep])
average_confidence = np.mean(np.max(predictions, axis=1))
print(f'Average prediction confidence: {average_confidence}')

# Generate confusion matrix and classification report
y_pred_classes = np.argmax(predictions, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
misclassified_indices = np.where(y_pred_classes != y_true_classes)[0]
print(f'Misclassified examples: {len(misclassified_indices)}')
print(f'Total examples: {len(y_test)}')

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true_classes, y_pred_classes))

# Plot Training and Validation Accuracy over Epochs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_final.history['accuracy'], label='Training Accuracy')
plt.plot(history_final.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot Training and Validation Loss over Epochs
plt.subplot(1, 2, 2)
plt.plot(history_final.history['loss'], label='Training Loss')
plt.plot(history_final.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# False Positive and False Negative Rates
false_positive = np.sum((y_pred_classes == 1) & (y_true_classes == 0))
false_negative = np.sum((y_pred_classes == 0) & (y_true_classes == 1))

print(f'False Positive Rate: {false_positive / np.sum(y_true_classes == 0)}')
print(f'False Negative Rate: {false_negative / np.sum(y_true_classes == 1)}')

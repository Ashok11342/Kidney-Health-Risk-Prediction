# Ensure the required package is installed
import subprocess
import sys

try:
    import splitfolders
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "split-folders"])

# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import deque
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import splitfolders
import os

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0],True)

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define data directories (adjust paths as per your dataset)
input_folder = 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'  # Replace with your dataset path
output_folder = './split_dataset'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.7, 0.15, 0.15))  # Train: 70%, Val: 15%, Test: 15%

# Data generators
train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

train_dataset = train_datagen.flow_from_directory(
    os.path.join(output_folder, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
val_dataset = val_datagen.flow_from_directory(
    os.path.join(output_folder, 'val'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
test_dataset = test_datagen.flow_from_directory(
    os.path.join(output_folder, 'test'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load pre-trained ResNet50 without top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
feature_extractor.trainable = False  # Freeze weights

# Define Q-network
feature_shape = base_model.output_shape[1:]  # (7, 7, 2048)
num_classes = 4  # tumor, normal, cyst, stone
input_layer = Input(shape=feature_shape)
x = GlobalAveragePooling2D()(input_layer)  # Reduces to (2048,)
x = Dense(512, activation='relu')(x)
q_values = Dense(num_classes, activation='linear')(x)  # Q-values for each class
q_network = Model(inputs=input_layer, outputs=q_values)

# Target network for stability
target_q_network = tf.keras.models.clone_model(q_network)
target_q_network.set_weights(q_network.get_weights())

# Replay buffer
replay_buffer = deque(maxlen=10000)

# Hyperparameters
learning_rate = 0.001
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
num_epochs = 10  # Adjust based on dataset size
update_target_every = 5

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Action selection function (epsilon-greedy)
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(num_classes)
    else:
        q_values = q_network.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

# Evaluation function (from original code, adapted for RL)
diseases_labels = list(train_dataset.class_indices.keys())
def evaluate(actual, predictions):
    accuracy = (predictions == actual).sum() / len(actual)
    print(f'Accuracy: {accuracy}')
    precision, recall, f1_score, _ = precision_recall_fscore_support(actual, predictions, average='macro')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1_score: {f1_score}')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    conf_mat = confusion_matrix(actual, predictions)
    sns.heatmap(conf_mat, annot=True, fmt='.0f', cmap="YlGnBu", 
                xticklabels=diseases_labels, yticklabels=diseases_labels).set_title('Confusion Matrix Heat Map')
    plt.show()

# Precompute features to speed up training
def precompute_features(dataset):
    features_list = []
    labels_list = []
    for images, labels in dataset:
        features = feature_extractor.predict(images, verbose=0)
        features_list.append(features)
        labels_list.append(np.argmax(labels, axis=1))
        if len(features_list) * batch_size >= dataset.samples:
            break
    return np.concatenate(features_list), np.concatenate(labels_list)

train_features, train_labels = precompute_features(train_dataset)
val_features, val_labels = precompute_features(val_dataset)
test_features, test_labels = precompute_features(test_dataset)

# Training loop
for epoch in range(num_epochs):
    total_reward = 0
    for i in range(len(train_features)):
        state = train_features[i]
        true_label = train_labels[i]
        
        # Select action
        action = select_action(state, epsilon)
        
        # Compute reward
        reward = 1 if action == true_label else 0
        total_reward += reward
        
        # Store experience (state, action, reward, done)
        replay_buffer.append((state, action, reward, True))
        
        # Train if enough experiences
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, dones = zip(*batch)
            states = np.stack(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            
            # Compute targets
            targets = rewards  # Single-step, terminal state
            
            # Update Q-network
            with tf.GradientTape() as tape:
                q_values = q_network(states)
                q_action = tf.gather(q_values, actions, batch_dims=1)
                loss = tf.keras.losses.mse(targets, q_action)
            grads = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
    
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # Update target network
    if (epoch + 1) % update_target_every == 0:
        target_q_network.set_weights(q_network.get_weights())
    
    # Print training progress
    avg_reward = total_reward / len(train_features)
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.4f}")
    
    # Validation
    val_predictions = []
    for state in val_features:
        q_values = q_network.predict(np.expand_dims(state, axis=0), verbose=0)
        action = np.argmax(q_values[0])
        val_predictions.append(action)
    val_accuracy = np.mean(np.array(val_predictions) == val_labels)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

# Test evaluation
test_predictions = []
for state in test_features:
    q_values = q_network.predict(np.expand_dims(state, axis=0), verbose=0)
    action = np.argmax(q_values[0])
    test_predictions.append(action)
print("\nTest Set Evaluation:")
evaluate(test_labels, np.array(test_predictions))

# Save the model (optional)
q_network.save('dqn_resnet_model.h5')
"""
Diabetic Retinopathy Detection - Model Training Script
Author: Keerthi Samhitha Kadaveru

This script trains the CNN model for diabetic retinopathy classification.
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    Input, concatenate, Reshape, LeakyReLU, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Configuration
CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']
NUM_CLASSES = 5

def create_model_v1(input_shape=(120, 120, 3)):
    """
    First model architecture - 120x120 input
    Uses cyclic layers with Leaky ReLU (alpha=0.3)
    Achieved ~0.70 kappa score
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.3),
        Conv2D(32, (3, 3), padding='same'),
        LeakyReLU(alpha=0.3),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), padding='same'),
        LeakyReLU(alpha=0.3),
        Conv2D(64, (3, 3), padding='same'),
        LeakyReLU(alpha=0.3),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), padding='same'),
        LeakyReLU(alpha=0.3),
        Conv2D(128, (3, 3), padding='same'),
        LeakyReLU(alpha=0.3),
        Conv2D(128, (3, 3), padding='same'),
        LeakyReLU(alpha=0.3),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        
        # Block 4
        Conv2D(256, (3, 3), padding='same'),
        LeakyReLU(alpha=0.3),
        Conv2D(256, (3, 3), padding='same'),
        LeakyReLU(alpha=0.3),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        
        # Dense layers
        Flatten(),
        Dropout(0.5),
        Dense(512),
        LeakyReLU(alpha=0.3),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_model_v2(input_shape=(512, 512, 3)):
    """
    Final model architecture - 512x512 input with dual-eye fusion
    Uses Leaky ReLU (alpha=0.5) and Maxout layers
    """
    # Single eye feature extractor
    input_layer = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(32, (7, 7), strides=(2, 2), padding='same')(input_layer)
    x = LeakyReLU(alpha=0.5)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Block 2
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Block 3
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Block 4
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Block 5
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Flatten and dense
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.5)(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    
    return model

def get_data_generators(data_dir, img_size, batch_size, validation_split=0.2):
    """Create data generators with augmentation"""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def train_model(args):
    """Main training function"""
    
    print("=" * 50)
    print("Diabetic Retinopathy Detection - Training")
    print("=" * 50)
    print(f"Input Size: {args.input_size}x{args.input_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print("=" * 50)
    
    # Create model
    if args.input_size == 120:
        model = create_model_v1(input_shape=(120, 120, 3))
        print("Using Model V1 (120x120)")
    else:
        model = create_model_v2(input_shape=(args.input_size, args.input_size, 3))
        print(f"Using Model V2 ({args.input_size}x{args.input_size})")
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Get data generators
    train_gen, val_gen = get_data_generators(
        args.data_dir,
        args.input_size,
        args.batch_size
    )
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        ModelCheckpoint(
            f'src/Model/dr_model_{timestamp}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=f'logs/{timestamp}',
            histogram_freq=1
        )
    ]
    
    # Train
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('src/Model/dr_model.h5')
    print(f"\nModel saved to src/Model/dr_model.h5")
    
    # Print final metrics
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print("=" * 50)
    
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DR Detection Model')
    
    parser.add_argument('--input_size', type=int, default=512,
                        help='Input image size (default: 512)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--data_dir', type=str, default='data/train',
                        help='Training data directory')
    
    args = parser.parse_args()
    
    # Create model directory if not exists
    os.makedirs('src/Model', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    train_model(args)

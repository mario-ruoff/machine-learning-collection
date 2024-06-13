import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np


im_rows = 32
im_cols = 32

im_shape = (im_rows,im_cols,1) 

test_features = np.load('test_features.npy') 

def plot_performance(hist):

    fig, axs = plt.subplots(ncols=2, figsize=(12,6))

    axs[0].plot(hist.history['loss'],label='Trainingsdaten')
    axs[0].plot(hist.history['val_loss'],label='Evaluierungsdaten')

    axs[0].set_title('Verlust des Modells',fontsize=18)
    axs[0].set_ylabel('Verlust',fontsize=15)
    axs[0].set_xlabel('Epoche t',fontsize=15)
    axs[0].legend()


    axs[1].plot(hist.history['accuracy'],label='Trainingsdaten')
    axs[1].plot(hist.history['val_accuracy'],label='Evaluierungsdaten')
    axs[1].legend()


    axs[1].set_title('Accuracy des Modells',fontsize=18)
    axs[1].set_ylabel('Accuracy',fontsize=15)
    axs[1].set_xlabel('Epoche t',fontsize=15)
    axs[1].legend()


def plot_images(features, labels=None, ncols=3, nrows=3):
    fig, axes = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flat):
        ax.imshow(features[i],cmap='Greys')
        if labels is not None:
            ax.set_title(labels[i])
    plt.tight_layout()    


def prepare_data(train_features, train_labels, validation_features, validation_labels, test_features): 
    # Reshape input
    train_features = train_features.transpose(0, 3, 1, 2)
    validation_features = validation_features.transpose(0, 3, 1, 2)
    test_features = test_features.transpose(0, 3, 1, 2)
    train_features = train_features / 255
    validation_features = validation_features / 255
    test_features = test_features / 255

    # Map string labels to integers
    unique_labels = np.unique(train_labels)  # Get unique labels
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}  # Create a mapping

    # Convert string labels to integer labels
    train_labels_indices = [label_to_index[label] for label in train_labels]
    validation_labels_indices = [label_to_index[label] for label in validation_labels]
    
    # Convert integer labels to a tensor
    train_labels_tensor = torch.tensor(train_labels_indices, dtype=torch.long)
    validation_labels_tensor = torch.tensor(validation_labels_indices, dtype=torch.long)

    return train_features, train_labels_tensor, validation_features, validation_labels_tensor, test_features, label_to_index
 

def save_predict(predict:np.ndarray):
    assert type(predict) == np.ndarray, "Wrong type, should be 'np.ndarray'!"
    assert np.shape(predict) == (6000,), "Wrong shape, should be '(6000,)'!"
    assert predict.dtype == 'int64', "Wrong data type, should be 'dtype('int64')'"
    
    print("All checks passed! Saving files.")
    
    with open('predictions.npy', 'wb') as f:
        np.save(f, predict)
    print('Saving completed')

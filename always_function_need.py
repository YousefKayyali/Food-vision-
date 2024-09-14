import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 
import shutil
import pandas as pd
import sklearn as skl
import always_function_need as afn
import random
from tqdm.notebook import tqdm
import tensorboard
from  sklearn.metrics import confusion_matrix
from sklearn.metrics  import recall_score
from sklearn.metrics  import precision_score
import itertools
import json
from sklearn.metrics  import f1_score
from tensorflow.keras import layers
def image_aug(img,label,scale=True):
    '''
    data augmantaion function for data set instance
    '''
    if scale:
        img=tf.keras.layers.Rescaling(1.0/255)(img)
    img=layers.RandomFlip()(img)
    img=layers.RandomBrightness(factor=0.6,value_range=(0,1))(img)
    img=layers.RandomRotation(factor=(-0.3,0.2))(img)
    img=layers.RandomZoom(height_factor=(-0.2,-0.3),width_factor=(-0.2,-0.3))(img)
    img=layers.RandomContrast(factor=1)(img)
    return img,label




def loss_accuracy_plot(history:tf.keras.callbacks.History):
    loss_traing=history.history['loss']
    loss_val=history.history['val_loss']
    accuracy=history.history['categorical_accuracy']
    accuracy_val=history.history['val_categorical_accuracy']
    epoch=history.epoch
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
   # plt.semilogy()
    plt.grid()
    plt.plot(epoch,loss_traing,label="traing loss")
    plt.plot(epoch,loss_val,label="val loss")
    plt.legend()
    plt.subplot(1,2,2)
  #  plt.semilogy()
    plt.grid()
    plt.plot(epoch,accuracy,label="traing acc")
    plt.plot(epoch,accuracy_val,label="val acc")
    plt.legend(title_fontsize="xx-large")
    plt.show()



def load_json(path):
    f=open(path)
    data= json.load(f)
    f.close()
    return data

def create_target_class_folder(parint_folder,new_subfolder,image_folder,data_set,target_varible,path):
    print(f"target data sets {data_set}")
    labels=load_json(path+'\\'+data_set+'.json')
    paths=(parint_folder+'\\'+new_subfolder+'\\'+data_set)
    for i in target_varible:
        os.makedirs(parint_folder+'\\'+new_subfolder+'\\'+data_set+'\\'+i,exist_ok=True)
        #movea image 
        images_moved = [] # Keep track of images moved
        for j in labels[i]:
            past_path = os.path.join(image_folder, j.replace('/', '\\'))+'.jpg'
            new_path = os.path.join(parint_folder, new_subfolder, data_set,i, os.path.basename(j.replace('/', '\\')))+'.jpg'
            shutil.copy2(past_path,new_path)
            images_moved.append(new_path)
        print(f"image moved from {data_set} of {i} is {len(images_moved)}")
    return paths
import datetime
def create_tensorbord_callback(dir_name, experiment):
    ''' 
    take the name path of dir to save it and the exprinment to name the model and save log in that file 
    
    
    '''
    log_dir = os.path.join(dir_name, experiment, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,write_graph=True,write_images=True,
    update_freq='epoch', )
    print(f"Created log dir: {log_dir}")
    return tensorboard_callback

def viwe_random_image_fdir(path,target):
  target_path=path+'\\'+target
  random_image=random.sample(os.listdir(target_path),k=1)
  img=plt.imread(target_path+'\\'+random_image[0],format='jpg')
  print(type(img))
  print(img.shape)
  plt.imshow(img)
  plt.axis(False)
  return img
def create_check_point_inst(expr:str,save_best=True):
    '''
    in this project folder we just take the expr name and not need to full path dir
    save best model by defualts you can change it 
    we just save wights not all model 
    '''
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=r'check_point\\'+expr,
    save_best_only=save_best,
    monitor='val_loss',
    verbose=0,
    save_weights_only=True
    )
    return checkpoint_callback
def get_percent_images(target_dir, new_dir, sample_amount=0.1, random_state=42):
    """
    Get sample_amount percentage of random images from target_dir and copy them to new_dir.
    
    Preserves subdirectory file names.
    
    E.g. target_dir=pizza_steak/train/steak/all_files 
                -> new_dir_name/train/steak/X_percent_of_all_files
                
    Parameters
    --------
    target_dir (str) - file path of directory you want to extract images from
    new_dir (str) - new directory path you want to copy original images to
    sample_amount (float), default 0.1 - percentage of images to copy (e.g. 0.1 = 10%)
    random_state (int), default 42 - random seed value 
    """
    # Set random seed for reproducibility
    random.seed(random_state)
    
    # Get a list of dictionaries of image files in target_dir
    # e.g. [{"class_name":["2348348.jpg", "2829119.jpg"]}]
    images = [{dir_name: os.listdir(target_dir+"\\" + dir_name)} for dir_name in os.listdir(target_dir)]

    for i in images:
        for k, v in i.items():
            # How many images to sample?
            sample_number = round(int(len(v)*sample_amount))
            print(f"There are {len(v)} total images in '{target_dir+k}' so we're going to copy {sample_number} to the new directory.")
            print(f"Getting {sample_number} random images for {k}...")
            random_images = random.sample(v, sample_number)

            # Make new dir for each key
            new_target_dir = new_dir + k
            print(f"Making dir: {new_target_dir}")
            os.makedirs(new_target_dir, exist_ok=True)

            # Keep track of images moved
            images_moved = []

            # Create file paths for original images and new file target
            print(f"Copying images from: {target_dir}\n\t\t to: {new_target_dir}/\n")
            for file_name in tqdm(random_images):
                og_path = target_dir+"\\" + k + "\\" + file_name
                new_path = new_target_dir + "\\" + file_name

                # Copy images from OG path to new path
                shutil.copy2(og_path, new_path)
                images_moved.append(new_path)

            # Make sure number of images moved is correct
            assert len(os.listdir(new_target_dir)) == sample_number
            assert len(images_moved) == sample_number



def comper_loss_accuracy_plot(history:tf.keras.callbacks.History,history_new:tf.keras.callbacks.History):
    loss_traing=history.history['loss']
    loss_val=history.history['val_loss']
    accuracy=history.history['categorical_accuracy']
    accuracy_val=history.history['val_categorical_accuracy']
    new_loss_traing=history_new.history['loss']
    new_loss_val=history_new.history['val_loss']
    new_accuracy=history_new.history['categorical_accuracy']
    new_accuracy_val=history_new.history['val_categorical_accuracy']
    total_loss=loss_traing+new_loss_traing
    total_loss_val=loss_val+new_loss_val
    total_acc=accuracy+new_accuracy
    total_acc_val=accuracy_val+new_accuracy_val
    epoch=history.epoch
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
   # plt.semilogy()
    plt.grid()
    plt.plot(total_acc,label="Training Accuracy")
    plt.plot(total_acc_val,label="Valdation Accuracy")
    plt.plot([4,4],plt.ylim(),label="start fun tuning")
    plt.legend()
    plt.subplot(1,2,2)
  #  plt.semilogy()
    plt.grid()
    plt.plot(total_loss,label="Training loss")
    plt.plot(total_loss_val,label="Valdation loss")
    plt.plot([4,4],plt.ylim(),label="start fun tuning")
    plt.legend()
    plt.show()
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    cm = confusion_matrix(y_true, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm[np.isnan(cm_norm)] = 0

    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm if not norm else cm_norm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    labels = classes if classes else np.arange(n_classes)
    labels = labels[:n_classes]

    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels
    )
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    plt.xticks(rotation=45, ha="right", fontsize=text_size)
    plt.yticks(fontsize=text_size)

    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)" if norm else f"{cm[i, j]}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)

    plt.tight_layout()
    if savefig:
        plt.savefig("confusion_matrix.png")

    plt.show()


def evluate_on_diff_metrix(true_labels,prdications):
    ''' 
    take true label and predication in form of probablites of each class.
    print:
    recall score
    perscion score
    F1 score
    '''
    true_labels = np.argmax(true_labels, axis=1)
    predicted_labels = np.argmax(prdications, axis=1) 
    print(f"Recall Score: {recall_score(true_labels, predicted_labels, average='macro')}")
    print(f"Perscion Score: {precision_score(true_labels,predicted_labels, average='macro')}")
    print(f"F1 Score: {f1_score(true_labels,predicted_labels, average='macro')}")
    
def score_classes(metrix_type:str,class_name,scores):
    plt.figure(figsize=(20,20))
    plt.barh(range(len(scores)),scores*100,height=0.7)
    plt.yticks(ticks=range(len(scores)),labels=class_name);
    plt.xlabel(f"{metrix_type} score")
    plt.title(f"{metrix_type} score for diff 101 class")
    for i, score in enumerate(scores):
        plt.text(score * 100 + 0.5, i, f"{score * 100:.1f}%", va='center', fontsize=10)
    plt.show()

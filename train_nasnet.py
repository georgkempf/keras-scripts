from keras.models import load_model
from keras.applications.nasnet import NASNetLarge
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import os
import argparse
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks
import re

class Config():
    def __init__(self, args):
        self.dataset_dir = args.dataset_dir
        self.train_dir = args.train_dir
        
    def get_dataset_dir(self):
        return self.dataset_dir
        
    def get_train_dir(self):
        return self.train_dir
    
    def set_last_checkpoint(self, last_checkpoint):
        self.last_checkpoint = last_checkpoint

class DatasetFactory():
    def get_datasets(self, config):
        
        dataset_path = config.get_dataset_dir()
        
        train_datagen = ImageDataGenerator(rescale=1./255)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
                os.path.join(dataset_path, 'train'),
                target_size=(250, 250),
                batch_size=32)
        
        validation_generator = test_datagen.flow_from_directory(
                os.path.join(dataset_path, 'validation'),
                target_size=(250, 250),
                batch_size=32)

        return train_generator, validation_generator
    
class LossHistory(keras.callbacks.Callback):
    def __init__(self, config):
        self.config = config
        
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.history = {}

    def on_batch_end(self, batch, logs={}):
        print("Log file content")
        print(logs)
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))
        self.history['acc'] = self.accuracy
        self.history['loss'] = self.losses
        self.history['val'] = False
        Visualize(self.history, self.config).plot_history()
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))
        self.history['acc'] = self.accuracy
        self.history['val_acc'] = self.val_accuracy
        self.history['loss'] = self.losses
        self.history['val_loss'] = self.val_losses
        self.history['val'] = True
        Visualize(self.history, self.config, epoch).plot_history()
        
        
class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, config):
        super(ModelCheckpoint, self).__init__()
        self.config = config
        
    def on_batch_end(self, batch, logs={}):
        self.model.save_weights(os.path.join(self.config.train_dir, 'checkpoint_{}.hdf5'.format(self.config.last_checkpoint)))
        self.config.set_last_checkpoint += 1
        
class Visualize():
        def __init__(self, history, config, epoch=None):
            self.history = history
            train_dir = config.get_train_dir()
            self.epoch = epoch
            if self.epoch is None:
                self.accuracy_plot_path = os.path.join(train_dir, 'accuracy.png')
                self.loss_plot_path = os.path.join(train_dir, 'loss.png')
            else:
                self.accuracy_plot_path = os.path.join(train_dir, 'accuracy_{}.png'.format(self.epoch))
                self.loss_plot_path = os.path.join(train_dir, 'loss_{}.png'.format(self.epoch))
            if os.path.exists(self.accuracy_plot_path):
                os.remove(self.accuracy_plot_path)
            if os.path.exists(self.loss_plot_path):
                os.remove(self.loss_plot_path)
            
        def plot_history(self):
            # Plot training & validation accuracy values
            plt.plot(self.history['acc'])
            if self.history['val']:
                plt.plot(self.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            
            if self.history['val']:
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
            else:
                plt.xlabel('Step')
                plt.legend(['Train'], loc='upper left')
            plt.savefig(self.accuracy_plot_path)
            
            # Plot training & validation loss values
            plt.plot(self.history['loss'])
            if self.history['val']:
                plt.plot(self.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            
            if self.history['val']:
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
            else:
                plt.xlabel('Step')
                plt.legend(['Train'], loc='upper left')
            plt.savefig(self.loss_plot_path)

class Train():
    def __init__(self, config):
        self.config = config
        self.train_dir = self.config.get_train_dir()
        self.model_path = os.path.join(self.train_dir, 'model.h5')
        self.model_weight_path = os.path.join(self.train_dir, 'model_weights.h5')
        self.base_model_path = os.path.join(self.train_dir, 'base_model.h5')
        self.train_generator, self.validation_generator = DatasetFactory().get_datasets(config)
        self.last_checkpoint, self.last_checkpoint_num = self.get_last_checkpoint()
        self.base_model = None
        if os.path.exists(self.base_model_path) and os.path.exists(self.model_path):
            print("Found model file!")
            self.model= load_model(self.model_path)
            if not self.last_checkpoint is None:
                print("Restoring from last checkpoint: %s" %self.last_checkpoint)
                try:
                    self.model.load_weights(self.last_checkpoint)
                except OSError:
                    self.last_checkpoint, _ = self.get_last_checkpoint(one_previous=True)
                    self.model.load_weights(self.last_checkpoint)
        else:
            self.model, self.base_model = self.add_top_layers()
            
    def get_last_checkpoint(self, one_previous=False):
        checkpoint_nums = []
        for file in os.listdir(self.train_dir):
            if re.match("checkpoint_\d+.hdf5", file):
                checkpoint_num = re.match("checkpoint_(\d+).hdf5", file).group(1)
                checkpoint_nums.append(int(checkpoint_num))
        if checkpoint_nums != []:
            print("Found checkpoint files")
            if not one_previous:
                last_checkpoint_num = max(checkpoint_nums)
            else:
                last_checkpoint_num = max(checkpoint_nums) - 1
            last_checkpoint = os.path.join(self.train_dir,'checkpoint_{}.hdf5'.format(last_checkpoint_num))
            if len(checkpoint_nums) > 6:
                checkpoint_nums = sorted(checkpoint_nums)
                to_delete = [x for x in checkpoint_nums[:-3]]
                for num in to_delete:
                    try:
                        os.remove(os.path.join(self.train_dir,'checkpoint_{}.hdf5'.format(num)))
                    except Exception as e:
                        pass
            return last_checkpoint, last_checkpoint_num
        else:
            return None, None

            
    def add_top_layers(self):
        base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(250,250,3))
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(2, activation='softmax')(x)
        
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        base_model.save(self.base_model_path)
        model.save(self.model_path)
        
        return model, base_model
    
    def print_layers(self):
        for i, layer in enumerate(self.model.layers):
           print(i, layer.name)
        
    
    def train_top_layers(self):
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        if not self.base_model is None:
            for layer in self.base_model.layers:
                layer.trainable = False
        else:
            for layer in self.model.layers[:-3]:
                layer.trainable = False
        
        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        
        checkpoint_path = os.path.join(self.train_dir, 'checkpoint.hdf5')
        csv_logger = keras.callbacks.CSVLogger(os.path.join(self.train_dir, 'training.log'), append=True)
        _, last_checkpoint = self.get_last_checkpoint()
        self.config.set_last_checkpoint(last_checkpoint)
        checkpointer = ModelCheckpoint(self.config)
        
        history = LossHistory(self.config)
        
        # train the model on the new data for a few epochs
        self.model.fit_generator(self.train_generator,
                steps_per_epoch=20,
                epochs=5,
                validation_data=self.validation_generator,
                validation_steps=20,
                callbacks=[checkpointer, history, csv_logger])
        
        self.model.save(self.model_path)
        self.model.save_weights(self.model_weight_path)
        
        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.
    
    def fine_tune(self):
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in self.model.layers[:249]:
           layer.trainable = False
        for layer in self.model.layers[249:]:
           layer.trainable = True
        
        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        from keras.optimizers import SGD
        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy')
        
        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        self. model.fit_generator(self.train_generator,
                steps_per_epoch=2000,
                epochs=50,
                validation_data=self.validation_generator,
                validation_steps=800)
        

parser = argparse.ArgumentParser(description='Train a keras model')
parser.add_argument('--dataset_dir', dest='dataset_dir', type=str, help='Base directory of data')
parser.add_argument('--train_dir', dest='train_dir', type=str, help='Directory where to save checkpoints and logs')
args = parser.parse_args()

config = Config(args)
Train(config).train_top_layers()

from tensorflow.keras.callbacks import Callback

class TrainingLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch +1}:")
        print(f"Train loss: {logs.get('loss')}")
        print(f"Train accuracy: {logs.get('accuracy')}")
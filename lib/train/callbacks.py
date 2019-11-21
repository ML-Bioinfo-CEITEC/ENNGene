import tensorflow


class ProgressMonitor(tensorflow.keras.callbacks.Callback):

    def __init__(self, epochs, progress_bar=None, progress_status=None, chart=None):
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.progress_status = progress_status
        self.chart = chart

    def on_epoch_begin(self, epoch, logs=None):
        self.progress_bar.progress(epoch/self.epochs)
        self.progress_status.text(f'Epoch {epoch}/{self.epochs}')

    def on_epoch_end(self, epoch, logs=None):
        epoch_data = {'Train loss': [logs['loss']],
                      'Train accuracy': [logs['accuracy']],
                      'Validation loss': [logs['val_loss']],
                      'Validation accuracy': [logs['val_accuracy']]}
        self.chart.add_rows(epoch_data)

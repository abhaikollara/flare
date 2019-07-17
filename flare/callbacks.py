
class Callback(object):


    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

    def on_eval_begin(self, logs={}):
        pass

    def on_eval_end(self, logs={}):
        pass

    def on_epoch_begin(self, logs={}):
        pass

    def on_epoch_end(self, logs={}):
        pass

    def on_train_batch_begin(self, logs={}):
        pass

    def on_train_batch_end(self, logs={}):
        pass

    def on_eval_batch_begin(self, logs={}):
        pass

    def on_eval_batch_end(self, logs={}):
        pass

class CallbackList(object):

    def __init__(self, callbacks=[]):
        self.callbacks = callbacks

    def on_train_begin(self, logs={}):
        for cbk in self.callbacks:
            cbk.on_train_begin(logs)

    def on_train_end(self, logs={}):
        for cbk in self.callbacks:
            cbk.on_train_end(logs)

    def on_eval_begin(self, logs={}):
        for cbk in self.callbacks:
            cbk.on_eval_begin(logs)

    def on_eval_end(self, logs={}):
        for cbk in self.callbacks:
            cbk.on_eval_end(logs)

    def on_epoch_begin(self, logs={}):
        for cbk in self.callbacks:
            cbk.on_epoch_begin(logs)

    def on_epoch_end(self, logs={}):
        for cbk in self.callbacks:
            cbk.on_epoch_end(logs)

    def on_train_batch_begin(self, logs={}):
        for cbk in self.callbacks:
            cbk.on_train_batch_begin(logs)

    def on_train_batch_end(self, logs={}):
        for cbk in self.callbacks:
            cbk.on_train_batch_end(logs)

    def on_eval_batch_begin(self, logs={}):
        for cbk in self.callbacks:
            cbk.on_eval_batch_begin(logs)

    def on_eval_batch_end(self, logs={}):
        for cbk in self.callbacks:
            cbk.on_eval_batch_end(logs)


class Baselogger(Callback):

    def __init__(self, metric=None):
        self.train_loss = 0.0
        self.test_loss = 0.0
        self.train_loss_history = []
        self.test_loss_history = []

        self.metric = metric
        self.metric_value = 0.0

    def on_train_begin(self, logs={}):
        print("Training")

    def on_eval_begin(self, logs={}):
        print("Evaluating")

    def on_epoch_begin(self, logs={}):
        print("Epoch", logs['epoch']+1)
    
    def on_train_batch_end(self, logs={}):
        batch_no = logs['batch_no']
        n_batches = logs['n_batches']
        batch_loss = logs['batch_loss']
        self.train_loss = ((self.train_loss * batch_no) + batch_loss)/ (batch_no + 1)
        if self.metric is not None:
            batch_metric_value = self.metric(logs['y'], logs['Y']).item()
            self.metric_value = ((self.metric_value * batch_no) + batch_metric_value)/ (batch_no + 1)

        if batch_no > 0:
            print("\r", end="")
        
        if self.metric is not None:
            print(f" Batch {batch_no+1}/{n_batches} Loss {round(self.train_loss,4)} {self.metric.name} {round(self.metric_value,4)}", end="")
        else:
            print(f" Batch {batch_no+1}/{n_batches} Loss {round(self.train_loss,4)}", end="")

    def on_eval_batch_end(self, logs={}):
        batch_no = logs['batch_no']
        n_batches = logs['n_batches']
        batch_loss = logs['batch_loss']
        self.test_loss = ((self.test_loss * batch_no) + batch_loss)/ (batch_no + 1)
        if self.metric is not None:
            batch_metric_value = self.metric(logs['y'], logs['Y']).item()
            self.metric_value = ((self.metric_value * batch_no) + batch_metric_value)/ (batch_no + 1)

        if batch_no > 0:
            print("\r", end="")

        if self.metric is not None:
            print(f" Batch {batch_no+1}/{n_batches} Loss {round(self.test_loss,4)} {self.metric.name} {round(self.metric_value,4)}", end="")
        else:
            print(f" Batch {batch_no+1}/{n_batches} Loss {round(self.test_loss,4)}", end="")

    def on_eval_end(self, logs={}):
        self.test_loss_history.append(self.test_loss)
        self.test_loss = 0.0
        self.metric_value = 0.0
        print("\n")

    def on_epoch_end(self, logs={}):
        self.train_loss_history.append(self.train_loss)
        self.train_loss = 0.0
        self.metric_value = 0.0
        print("\n")

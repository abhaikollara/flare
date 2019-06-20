
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


class LossHistory(Callback):

    def __init__(self):
        self.train_loss = 0.0
        self.test_loss = 0.0

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

        if batch_no > 0:
            print("\r", end="")
        print(f" Batch {batch_no+1}/{n_batches} Loss {round(self.train_loss,4)}", end="")
    
    def on_eval_batch_end(self, logs={}):
        batch_no = logs['batch_no']
        n_batches = logs['n_batches']
        batch_loss = logs['batch_loss']

        self.test_loss = ((self.test_loss * batch_no) + batch_loss)/ (batch_no + 1)

        if batch_no > 0:
            print("\r", end="")
        print(f" Batch {batch_no+1}/{n_batches} Loss {round(self.test_loss,4)}", end="")
    
    def on_eval_end(self, logs={}):
        print("\n")

    def on_epoch_end(self, logs={}):
        print("\n")


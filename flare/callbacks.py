
class Callback(object):

    def set_trainer(self, trainer):
        self.trainer = trainer

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

    def __init__(self, trainer, callbacks=[]):
        self.callbacks = callbacks
        for callback in callbacks:
            callback.set_trainer(trainer)

    def append(self, metric):
        self.callbacks.append(metric)

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

        if batch_no > 0:
            print("\r", end="")
        
        console_text = f" Batch {batch_no+1}/{n_batches} Loss {round(self.train_loss,4)}"
        if self.metric is not None:
            metric_logs = [f"{m.name} {round(m.value, 4)}" for m in self.trainer.metrics]
            console_text = " ".join([console_text, *metric_logs])

        print(console_text, end="")

    def on_eval_batch_end(self, logs={}):
        batch_no = logs['batch_no']
        n_batches = logs['n_batches']
        batch_loss = logs['batch_loss']
        self.test_loss = ((self.test_loss * batch_no) + batch_loss)/ (batch_no + 1)

        if batch_no > 0:
            print("\r", end="")
        
        console_text = f" Batch {batch_no+1}/{n_batches} Loss {round(self.test_loss,4)}"
        if self.metric is not None:
            metric_logs = [f"{m.name} {round(m.value, 4)}" for m in self.trainer.metrics]
            console_text = " ".join([console_text, *metric_logs])

            # print(f" Batch {batch_no+1}/{n_batches} Loss {round(self.test_loss,4)} {self.metric.name} {round(self.metric_value,4)}", end="")
        
        print(console_text, end="")

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


class MetricLogger(Callback):

    def __init__(self, metrics):
        self.metrics = metrics
        self.train_logs = {str(m):[] for m in self.metrics}
        self.test_logs = {str(m):[] for m in self.metrics}

    def on_train_begin(self, logs={}):
        for m in self.metrics:
            m.reset_states()

    def on_eval_begin(self, logs={}):
        for m in self.metrics:
            m.reset_states()

    def on_eval_end(self, logs={}):
        for m in self.metrics:
            self.test_logs[str(m)].append(m.value)

    def on_epoch_begin(self, logs={}):
        for m in self.metrics:
            m.reset_states()

    def on_epoch_end(self, logs={}):
        for m in self.metrics:
            self.train_logs[str(m)].append(m.value)

    def on_train_batch_end(self, logs={}):
        for m in self.metrics:
            m.batch_update(logs['batch_no'], logs['y'], logs['Y'])

    def on_eval_batch_end(self, logs={}):
        for m in self.metrics:
            m.batch_update(logs['batch_no'], logs['y'], logs['Y'])
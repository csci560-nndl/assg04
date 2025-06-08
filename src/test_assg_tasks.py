import numpy as np
import pandas as pd
import sklearn
import unittest
from tensorflow import keras
import keras.src
from tensorflow.keras import layers
#from twisted.trial import unittest
from unittest.mock import patch
from io import StringIO
#from assg_tasks import EveryNEpochs
#from assg_tasks import get_unregularized_model
#from assg_tasks import get_l2_regularized_model
#from assg_tasks import get_dropout_model


class test_EveryNEpochs_callback(unittest.TestCase):
    def setUp(self):
        self.c = EveryNEpochs(16)
        self.c.params = {'epochs': 100}
        self.logs = {
            'loss': 0.1234,
            'accuracy': 0.5678,
            'val_loss': 0.4321,
            'val_accuracy': 0.8765        
        }
        pass

    def test_callback_creation(self):
        self.assertEqual(self.c.N, 16)
        self.assertIsInstance(self.c, EveryNEpochs)

    def test_epoch_5_no_output(self):
        # no output expected at epoch 5
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.c.on_epoch_end(5, self.logs)
            self.assertEqual(fake_out.getvalue(), "")

    def test_epoch_0_output(self):
        # should always get output for epoch 0
        expected_out = "Epoch: 0000/0099  training loss - 0.1234  validation loss - 0.4321  training_accuracy - 0.5678  validation accuracy - 0.8765\n"
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.c.on_epoch_end(0, self.logs)
            self.assertEqual(fake_out.getvalue(), expected_out)

    def test_epoch_16_output(self):
        # with N=16 should get output at epoch 16
        expected_out = "Epoch: 0016/0099  training loss - 0.1234  validation loss - 0.4321  training_accuracy - 0.5678  validation accuracy - 0.8765\n"
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.c.on_epoch_end(16, self.logs)
            self.assertEqual(fake_out.getvalue(), expected_out)

    def test_epoch_99_output(self):
        # with epochs=100 last epoch is 99, should always get output on last epoch
        expected_out = "Epoch: 0099/0099  training loss - 0.1234  validation loss - 0.4321  training_accuracy - 0.5678  validation accuracy - 0.8765\n"
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.c.on_epoch_end(99, self.logs)
            self.assertEqual(fake_out.getvalue(), expected_out)


class test_get_unregularized_model(unittest.TestCase):

    def setUp(self):
        self.model = get_unregularized_model()

    def test_model_layers(self):
        self.assertEqual(len(self.model.layers), 5)
        self.assertIsInstance(self.model.layers[0], keras.src.layers.core.input_layer.InputLayer)
        self.assertEqual(self.model.layers[0].batch_shape, (None, 2))

        self.assertIsInstance(self.model.layers[1], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.model.layers[1].weights[0].shape, (2,1024))
        self.assertEqual(self.model.layers[1].activation, keras.src.activations.activations.relu)

        self.assertIsInstance(self.model.layers[2], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.model.layers[2].weights[0].shape, (1024,512))
        self.assertEqual(self.model.layers[2].activation, keras.src.activations.activations.relu)

        self.assertIsInstance(self.model.layers[3], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.model.layers[3].weights[0].shape, (512,128))
        self.assertEqual(self.model.layers[3].activation, keras.src.activations.activations.relu)

        self.assertIsInstance(self.model.layers[4], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.model.layers[4].weights[0].shape, (128,1))
        self.assertEqual(self.model.layers[4].activation, keras.src.activations.activations.sigmoid)

    def test_model_attributes(self):
        self.assertEqual(self.model.loss, 'binary_crossentropy')
        self.assertIsInstance(self.model.optimizer, keras.src.optimizers.rmsprop.RMSprop)
        self.assertEqual(self.model.metrics_names[0], 'loss')
        # would like to check the other metric is accuracy, but seems hidden?...
        self.assertEqual(self.model.metrics_names[1], 'compile_metrics')

    def test_model_names(self):
        self.assertEqual(self.model.name, 'unregularized_model')
        self.assertEqual(self.model.layers[0].name, 'input')
        self.assertEqual(self.model.layers[1].name, 'hidden1')
        self.assertEqual(self.model.layers[2].name, 'hidden2')
        self.assertEqual(self.model.layers[3].name, 'hidden3')        
        self.assertEqual(self.model.layers[4].name, 'output')


# note basically duplicate of tests, just add in check that
# asked for regularization is being used
class test_get_l2_regularized_model(unittest.TestCase):

    def setUp(self):
        self.model = get_l2_regularized_model(0.002)

    def test_model_layers(self):
        self.assertEqual(len(self.model.layers), 5)
        self.assertIsInstance(self.model.layers[0], keras.src.layers.core.input_layer.InputLayer)
        self.assertEqual(self.model.layers[0].batch_shape, (None, 2))

        self.assertIsInstance(self.model.layers[1], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.model.layers[1].weights[0].shape, (2,1024))
        self.assertEqual(self.model.layers[1].activation, keras.src.activations.activations.relu)
        self.assertIsInstance(self.model.layers[1].kernel_regularizer, keras.src.regularizers.regularizers.L2)
        self.assertEqual(self.model.layers[1].kernel_regularizer.l2, 0.002)

        self.assertIsInstance(self.model.layers[2], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.model.layers[2].weights[0].shape, (1024,512))
        self.assertEqual(self.model.layers[2].activation, keras.src.activations.activations.relu)
        self.assertIsInstance(self.model.layers[2].kernel_regularizer, keras.src.regularizers.regularizers.L2)
        self.assertEqual(self.model.layers[2].kernel_regularizer.l2, 0.002)

        self.assertIsInstance(self.model.layers[3], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.model.layers[3].weights[0].shape, (512,128))
        self.assertEqual(self.model.layers[3].activation, keras.src.activations.activations.relu)
        self.assertIsInstance(self.model.layers[3].kernel_regularizer, keras.src.regularizers.regularizers.L2)
        self.assertEqual(self.model.layers[2].kernel_regularizer.l2, 0.002)

        self.assertIsInstance(self.model.layers[4], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.model.layers[4].weights[0].shape, (128,1))
        self.assertEqual(self.model.layers[4].activation, keras.src.activations.activations.sigmoid)

    def test_model_attributes(self):
        self.assertEqual(self.model.loss, 'binary_crossentropy')
        self.assertIsInstance(self.model.optimizer, keras.src.optimizers.rmsprop.RMSprop)
        self.assertEqual(self.model.metrics_names[0], 'loss')
        # would like to check the other metric is accuracy, but seems hidden?...
        self.assertEqual(self.model.metrics_names[1], 'compile_metrics')

    def test_model_names(self):
        self.assertEqual(self.model.name, 'unregularized_model')
        self.assertEqual(self.model.layers[0].name, 'input')
        self.assertEqual(self.model.layers[1].name, 'hidden1')
        self.assertEqual(self.model.layers[2].name, 'hidden2')
        self.assertEqual(self.model.layers[3].name, 'hidden3')        
        self.assertEqual(self.model.layers[4].name, 'output')


class test_get_dropout_model(unittest.TestCase):

    def setUp(self):
        self.model = get_dropout_model(0.5)

    def test_model_layers(self):
        self.assertEqual(len(self.model.layers), 8)
        self.assertIsInstance(self.model.layers[0], keras.src.layers.core.input_layer.InputLayer)
        self.assertEqual(self.model.layers[0].batch_shape, (None, 2))

        self.assertIsInstance(self.model.layers[1], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.model.layers[1].weights[0].shape, (2,1024))
        self.assertEqual(self.model.layers[1].activation, keras.src.activations.activations.relu)

        self.assertIsInstance(self.model.layers[2], keras.src.layers.regularization.dropout.Dropout)
        self.assertEqual(self.model.layers[2].rate, 0.5)
        self.assertEqual(self.model.layers[2].output.shape, (None,1024))

        self.assertIsInstance(self.model.layers[3], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.model.layers[3].weights[0].shape, (1024,512))
        self.assertEqual(self.model.layers[3].activation, keras.src.activations.activations.relu)

        self.assertIsInstance(self.model.layers[4], keras.src.layers.regularization.dropout.Dropout)
        self.assertEqual(self.model.layers[4].rate, 0.5)
        self.assertEqual(self.model.layers[4].output.shape, (None,512))

        self.assertIsInstance(self.model.layers[5], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.model.layers[5].weights[0].shape, (512,128))
        self.assertEqual(self.model.layers[5].activation, keras.src.activations.activations.relu)

        self.assertIsInstance(self.model.layers[6], keras.src.layers.regularization.dropout.Dropout)
        self.assertEqual(self.model.layers[6].rate, 0.5)
        self.assertEqual(self.model.layers[6].output.shape, (None,128))

        self.assertIsInstance(self.model.layers[7], keras.src.layers.core.dense.Dense)
        self.assertEqual(self.model.layers[7].weights[0].shape, (128,1))
        self.assertEqual(self.model.layers[7].activation, keras.src.activations.activations.sigmoid)

    def test_model_attributes(self):
        self.assertEqual(self.model.loss, 'binary_crossentropy')
        self.assertIsInstance(self.model.optimizer, keras.src.optimizers.rmsprop.RMSprop)
        self.assertEqual(self.model.metrics_names[0], 'loss')
        # would like to check the other metric is accuracy, but seems hidden?...
        self.assertEqual(self.model.metrics_names[1], 'compile_metrics')

    def test_model_names(self):
        self.assertEqual(self.model.name, 'unregularized_model')
        self.assertEqual(self.model.layers[0].name, 'input')
        self.assertEqual(self.model.layers[1].name, 'hidden1')
        self.assertEqual(self.model.layers[2].name, 'dropout1')
        self.assertEqual(self.model.layers[3].name, 'hidden2')
        self.assertEqual(self.model.layers[4].name, 'dropout2')
        self.assertEqual(self.model.layers[5].name, 'hidden3')        
        self.assertEqual(self.model.layers[6].name, 'dropout3')
        self.assertEqual(self.model.layers[7].name, 'output')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
    This file aimes to build a multi-fidelity NN with ONE low fidelity and TWO high fidelity datasets
"""

from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config
from ...backend import tf
from ...utils import timing

class MfNN_LH2(NN):             ### Changed "Map" to "NN" for the new verison of deepXDE
    """Multifidelity neural networks.
    """

    def __init__(
        self,
        layer_size_low_fidelity,

        # layer_size_high_fidelity,
        layer_size_high_one_fidelity,     ##### Add two HIGH fidelity datasets
        layer_size_high_two_fidelity,     ##### Add two HIGH fidelity datasets

        activation,
        kernel_initializer,
        regularization=None,
        residue=False,
        trainable_low_fidelity=True,

        # trainable_high_fidelity=True,
        trainable_high_one_fidelity=True,     ##### Add two HIGH fidelity datasets
        trainable_high_two_fidelity=True,     ##### Add two HIGH fidelity datasets
    ):
        super(MfNN_LH2, self).__init__()
        self.layer_size_lo = layer_size_low_fidelity

        # self.layer_size_hi = layer_size_high_fidelity
        self.layer_size_hi_one = layer_size_high_one_fidelity     ##### Add two HIGH fidelity datasets
        self.layer_size_hi_two = layer_size_high_two_fidelity     ##### Add two HIGH fidelity datasets

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)
        self.residue = residue
        self.trainable_lo = trainable_low_fidelity

        # self.trainable_hi = trainable_high_fidelity
        self.trainable_hi_one = trainable_high_one_fidelity     ##### Add two HIGH fidelity datasets
        self.trainable_hi_two = trainable_high_two_fidelity     ##### Add two HIGH fidelity datasets


    @property
    def inputs(self):
        return self.X

    @property
    def outputs(self):
        return [self.y_lo, self.y_hi_one, self.y_hi_two]

    @property
    def targets(self):
        return [self.target_lo, self.target_hi_one, self.target_hi_two]

    @timing
    def build(self):
        print("Building multifidelity neural network...")
        self.X = tf.placeholder(config.real(tf), [None, self.layer_size_lo[0]])  

        #####--------------------------------------------------------------------
        ##### Build MfNN with 2 low fidelity layers and 1 high fidelity layers.
        #####--------------------------------------------------------------------

        # Low fidelity
        y = self.X
        for i in range(len(self.layer_size_lo) - 2):
            y = self.dense(
                y,
                self.layer_size_lo[i + 1],
                activation=self.activation,
                regularizer=self.regularizer,
                trainable=self.trainable_lo,
            )
        self.y_lo = self.dense(
            y,
            self.layer_size_lo[-1],
            regularizer=self.regularizer,
            trainable=self.trainable_lo,
        )

        #------------------------------------------------- 
        ##### Build two High fidelit NNs
        #------------------------------------------------- 

        ### Build the high_one NN
        X_hi = tf.concat([self.X, self.y_lo], 1)
        # Linear
        y_hi_one_l = self.dense(X_hi, self.layer_size_hi_one[-1], trainable=self.trainable_hi_one)
        # Nonlinear
        y = X_hi
        for i in range(len(self.layer_size_hi_one) - 1):
            y = self.dense(
                y,
                self.layer_size_hi_one[i],
                activation=self.activation,
                regularizer=self.regularizer,
                trainable=self.trainable_hi_one,
            )
        y_hi_one_nl = self.dense(
            y,
            self.layer_size_hi_one[-1],
            use_bias=False,
            regularizer=self.regularizer,
            trainable=self.trainable_hi_one,
        )
        # Linear + nonlinear
        if not self.residue:
            alpha = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi_one)
            alpha = activations.get("tanh")(alpha)
            self.y_hi_one = y_hi_one_l + alpha * y_hi_one_nl
        else:
            alpha1 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi_one)
            alpha1 = activations.get("tanh")(alpha1)
            alpha2 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi_one)
            alpha2 = activations.get("tanh")(alpha2)
            alpha3 = tf.Variable(1, dtype=config.real(tf), trainable=True)
           
            self.y_hi_one = alpha3 * self.y_lo + + 0.1 * (alpha1 * y_hi_one_l + alpha2 * y_hi_one_nl)


        ### Build the high_two NN
        X_hi = tf.concat([self.X, self.y_lo], 1)
        # Linear
        y_hi_two_l = self.dense(X_hi, self.layer_size_hi_two[-1], trainable=self.trainable_hi_two)
        # Nonlinear
        y = X_hi
        for i in range(len(self.layer_size_hi_two) - 1):
            y = self.dense(
                y,
                self.layer_size_hi_two[i],
                activation=self.activation,
                regularizer=self.regularizer,
                trainable=self.trainable_hi_two,
            )
        y_hi_two_nl = self.dense(
            y,
            self.layer_size_hi_two[-1],
            use_bias=False,
            regularizer=self.regularizer,
            trainable=self.trainable_hi_two,
        )
        # Linear + nonlinear
        if not self.residue:
            alpha = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi_two)
            alpha = activations.get("tanh")(alpha)
            self.y_hi_two = y_hi_two_l + alpha * y_hi_two_nl
        else:
            alpha4 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi_two)
            alpha4 = activations.get("tanh")(alpha1)
            alpha5 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi_two)
            alpha5 = activations.get("tanh")(alpha2)
            alpha6 = tf.Variable(1, dtype=config.real(tf), trainable=True)
           
            self.y_hi_two = alpha6 * self.y_lo + + 0.1 * (alpha4 * y_hi_two_l + alpha5 * y_hi_two_nl)



        self.target_lo = tf.placeholder(config.real(tf), [None, self.layer_size_lo[-1]]) 

        self.target_hi_one = tf.placeholder(config.real(tf), [None, self.layer_size_hi_one[-1]])     ##### Add two HIGH fidelity datasets
        self.target_hi_two = tf.placeholder(config.real(tf), [None, self.layer_size_hi_two[-1]])     ##### Add two HIGH fidelity datasets

        self.built = True


    def dense(
        self,
        inputs,
        units,
        activation=None,
        use_bias=True,
        regularizer=None,
        trainable=True,
    ):
        return tf.layers.dense(
            inputs,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=regularizer,
            trainable=trainable,
        )

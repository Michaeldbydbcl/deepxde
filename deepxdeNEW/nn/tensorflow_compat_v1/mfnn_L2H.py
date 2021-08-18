from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
    This file aimes to build a multi-fidelity NN with TWO low fidelity and ONE high fidelity datasets
"""

from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config
from ...backend import tf
from ...utils import timing

class MfNN_L2H(NN):             ### Changed "Map" to "NN" for the new verison of deepXDE
    """Multifidelity neural networks.
    """

    def __init__(
        self,
        # The previous two layers fidelity NN
        # layer_size_low_fidelity,
        # layer_size_high_fidelity,

        layer_size_low_one_fidelity, ### Add two low fidelity datasets
        layer_size_low_two_fidelity, ### Add two low fidelity datasets

        layer_size_high_fidelity,

        activation,
        kernel_initializer,
        regularization=None,
        residue=False,
        # trainable_low_fidelity=True,
        # trainable_high_fidelity=True,
        trainable_low_one_fidelity=True, ### Two low fidelity layers
        trainable_low_two_fidelity=True, ### Two low fidelity layers

        trainable_high_fidelity=True,

    ):
        super(MfNN_L2H, self).__init__()
        # self.layer_size_lo = layer_size_low_fidelity
        # self.layer_size_hi = layer_size_high_fidelity

        self.layer_size_lo_one = layer_size_low_one_fidelity ### Add layers for two low fidelty layers
        self.layer_size_lo_two = layer_size_low_two_fidelity ### Add layers for two low fidelty layers

        self.layer_size_hi = layer_size_high_fidelity

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)
        self.residue = residue
        # self.trainable_lo = trainable_low_fidelity
        # self.trainable_hi = trainable_high_fidelity

        self.trainable_lo_one = trainable_low_one_fidelity ### Two fidelity networks
        self.trainable_lo_two = trainable_low_two_fidelity ### Two fidelity networks

        self.trainable_hi = trainable_high_fidelity


    @property
    def inputs(self):
        return self.X

    @property
    def outputs(self):
        return [self.y_lo_one, self.y_lo_two, self.y_hi]

    @property
    def targets(self):
        return [self.target_lo_one, self.target_lo_two, self.target_hi]

    @timing
    def build(self):
        print("Building multifidelity neural network...")
        self.X = tf.placeholder(config.real(tf), [None, self.layer_size_lo_one[0]])  

        #####--------------------------------------------------------------------
        ##### Build MfNN with 2 low fidelity layers and 1 high fidelity layers.
        #####--------------------------------------------------------------------

        # Low fidelity
        # y = self.X
        # for i in range(len(self.layer_size_lo) - 2):
        #     y = self.dense(
        #         y,
        #         self.layer_size_lo[i + 1],
        #         activation=self.activation,
        #         regularizer=self.regularizer,
        #         trainable=self.trainable_lo,
        #     )
        # self.y_lo = self.dense(
        #     y,
        #     self.layer_size_lo[-1],
        #     regularizer=self.regularizer,
        #     trainable=self.trainable_lo,
        # )

        ###-------------------------------
        ### Build 2 low fidelity layers 

        y = self.X
        for i in range(len(self.layer_size_lo_one) - 2):
            y = self.dense(
                y,
                self.layer_size_lo_one[i + 1],
                activation=self.activation,
                regularizer=self.regularizer,
                trainable=self.trainable_lo_one,
            )
        self.y_lo_one = self.dense(
            y,
            self.layer_size_lo_one[-1],
            regularizer=self.regularizer,
            trainable=self.trainable_lo_one,
        )

        y = self.X
        for i in range(len(self.layer_size_lo_two) - 2):
            y = self.dense(
                y,
                self.layer_size_lo_two[i + 1],
                activation=self.activation,
                regularizer=self.regularizer,
                trainable=self.trainable_lo_two,
            )
        self.y_lo_two = self.dense(
            y,
            self.layer_size_lo_two[-1],
            regularizer=self.regularizer,
            trainable=self.trainable_lo_two,
        )


        # High fidelity
        X_hi = tf.concat([self.X, self.y_lo_one, self.y_lo_two], 1)
        # Linear
        y_hi_l = self.dense(X_hi, self.layer_size_hi[-1], trainable=self.trainable_hi)
        # Nonlinear
        y = X_hi
        for i in range(len(self.layer_size_hi) - 1):
            y = self.dense(
                y,
                self.layer_size_hi[i],
                activation=self.activation,
                regularizer=self.regularizer,
                trainable=self.trainable_hi,
            )
        y_hi_nl = self.dense(
            y,
            self.layer_size_hi[-1],
            use_bias=False,
            regularizer=self.regularizer,
            trainable=self.trainable_hi,
        )
        # Linear + nonlinear
        if not self.residue:
            alpha = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi)
            alpha = activations.get("tanh")(alpha)
            self.y_hi = y_hi_l + alpha * y_hi_nl
        else:
            alpha1 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi)
            alpha1 = activations.get("tanh")(alpha1)
            alpha2 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi)
            alpha2 = activations.get("tanh")(alpha2)

            alpha3 = tf.Variable(1, dtype=config.real(tf), trainable=True)
            alpha4 = tf.Variable(1, dtype=config.real(tf), trainable=True)
           
            self.y_hi = alpha3 * self.y_lo_one + alpha4 * self.y_lo_two + 0.1 * (alpha1 * y_hi_l + alpha2 * y_hi_nl)

        self.target_lo_one = tf.placeholder(config.real(tf), [None, self.layer_size_lo_one[-1]])   ### Two low fidelity layers
        self.target_lo_two = tf.placeholder(config.real(tf), [None, self.layer_size_lo_two[-1]])   ### Two low fidelity layers

        self.target_hi = tf.placeholder(config.real(tf), [None, self.layer_size_hi[-1]])

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

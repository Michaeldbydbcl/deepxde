from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
    This file aimes to build a multi-fidelity NN with ONE low fidelity, middle fidelity and high fidelity datasets
"""

from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config
from ...backend import tf
from ...utils import timing

class MfNN_LMH(NN):             ### Changed "Map" to "NN" for the new verison of deepXDE
    """Multifidelity neural networks.
    """

    def __init__(
        self,
        layer_size_low_fidelity,

        layer_size_mid_fidelity,      ##### Add MIDDLE fidelity datasets
        layer_size_high_fidelity,     ##### Add HIGH fidelity datasets

        activation,
        kernel_initializer,
        regularization=None,
        residue=False,
        trainable_low_fidelity=True,

        trainable_mid_fidelity=True,      ##### Add MIDDLE fidelity datasets
        trainable_high_fidelity=True,     ##### Add HIGH fidelity datasets
    ):
        super(MfNN_LMH, self).__init__()
        self.layer_size_lo = layer_size_low_fidelity

        self.layer_size_mi = layer_size_mid_fidelity     ##### Add MIDDLE fidelity datasets
        self.layer_size_hi = layer_size_high_fidelity     ##### Add HIGH fidelity datasets

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)
        self.residue = residue
        self.trainable_lo = trainable_low_fidelity

        self.trainable_mi = trainable_mid_fidelity      ##### Add MIDDLE fidelity datasets
        self.trainable_hi = trainable_high_fidelity     ##### Add HIGH fidelity datasets


    @property
    def inputs(self):
        return self.X

    @property
    def outputs(self):
        return [self.y_lo, self.y_mi, self.y_hi]

    @property
    def targets(self):
        return [self.target_lo, self.target_mi, self.target_hi]

    @timing
    def build(self):
        print("Building multifidelity neural network...")
        self.X = tf.placeholder(config.real(tf), [None, self.layer_size_lo[0]])  

        #####--------------------------------------------------------------------
        ##### Build MfNN with 1 low fidelity, 1 middle fidelity and 1 high fidelity layers.
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
        ##### Build Middle and High fidelitt NNs
        #------------------------------------------------- 

        ### Build the Middle NN
        X_mi = tf.concat([self.X, self.y_lo], 1)
        # Linear
        y_mi_l = self.dense(X_mi, self.layer_size_mi[-1], trainable=self.trainable_mi)
        # Nonlinear
        y = X_mi
        for i in range(len(self.layer_size_mi) - 1):
            y = self.dense(
                y,
                self.layer_size_mi[i],
                activation=self.activation,
                regularizer=self.regularizer,
                trainable=self.trainable_mi,
            )
        y_mi_nl = self.dense(
            y,
            self.layer_size_mi[-1],
            use_bias=False,
            regularizer=self.regularizer,
            trainable=self.trainable_mi,
        )
        # Linear + nonlinear
        if not self.residue:
            alpha = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_mi)
            alpha = activations.get("tanh")(alpha)
            self.y_mi = y_mi_l + alpha * y_mi_nl
        else:
            alpha1 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_mi)
            alpha1 = activations.get("tanh")(alpha1)
            alpha2 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_mi)
            alpha2 = activations.get("tanh")(alpha2)
            alpha3 = tf.Variable(1, dtype=config.real(tf), trainable=True)
           
            self.y_mi = alpha3 * self.y_lo + + 0.1 * (alpha1 * y_mi_l + alpha2 * y_mi_nl)


        ### Build the high NN 
        X_hi = tf.concat([self.X, self.y_mi], 1)      ##### !!! How to differentiate from LH2 NN? Feed y_mi to high NN?
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
            alpha4 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi)
            alpha4 = activations.get("tanh")(alpha4)
            alpha5 = tf.Variable(0, dtype=config.real(tf), trainable=self.trainable_hi)
            alpha5 = activations.get("tanh")(alpha5)

            alpha6 = tf.Variable(1, dtype=config.real(tf), trainable=True)
            alpha7 = tf.Variable(1, dtype=config.real(tf), trainable=True)
          
            self.y_hi = alpha6 * self.y_mi + alpha7 * self.y_lo + 0.1 * (alpha4 * y_hi_l + alpha5 * y_hi_nl)     #### add low or mid or both?
            # self.y_hi = alpha7 * self.y_lo + alpha6* self.y_mi + + 0.1 * (alpha4 * y_hi_l + alpha5 * y_hi_nl)     #### add low or mid or both?



        self.target_lo = tf.placeholder(config.real(tf), [None, self.layer_size_lo[-1]]) 

        self.target_mi = tf.placeholder(config.real(tf), [None, self.layer_size_mi[-1]])     ##### Add Middle fidelity datasets
        self.target_hi = tf.placeholder(config.real(tf), [None, self.layer_size_hi[-1]])     ##### Add HIGH fidelity datasets

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

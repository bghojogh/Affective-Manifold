# https://github.com/taki0112/ResNet-Tensorflow

import ops_resnet
import tensorflow as tf
import dataset_characteristics
import numpy as np
import Utils_losses


class ResNet_Siamese(object):

    def __init__(self, loss_type, feature_space_dimension, n_classes, n_samples_per_class_in_batch, n_res_blocks=18, margin_in_loss=0.25, batch_size=32):
        self.img_size_height = dataset_characteristics.get_image_height()
        self.img_size_width = dataset_characteristics.get_image_width()
        self.img_n_channels = dataset_characteristics.get_image_n_channels()
        self.c_dim = 3
        self.res_n = n_res_blocks
        self.feature_space_dimension = feature_space_dimension
        self.margin_in_loss = margin_in_loss
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_samples_per_class_in_batch = n_samples_per_class_in_batch

        self.x1 = tf.placeholder(tf.float32, [None, self.img_size_height, self.img_size_width, self.img_n_channels])
        self.x1Image = self.x1
        self.labels1 = tf.placeholder(tf.int32, [None,])

        self.is_first_batch = True
        self.covariance_prior = tf.placeholder(tf.float32, [self.feature_space_dimension, self.feature_space_dimension, self.n_classes])
        self.mean_prior = tf.placeholder(tf.float32, [1, self.feature_space_dimension, self.n_classes])
        self.n_samples_per_class_so_far = 0
        self.is_train = tf.placeholder(tf.int32)

        self.loss_type = loss_type
        # Create loss
        if self.is_train == 1:
            with tf.variable_scope("siamese") as scope:
                self.o1 = self.network(self.x1Image, is_training=True, reuse=False)
        else:
            with tf.variable_scope("siamese") as scope:
                self.o1 = self.network(self.x1Image, is_training=False, reuse=tf.AUTO_REUSE)
        if self.loss_type == "batch_hard_triplet":
            self.loss = self.batch_hard_triplet_loss(labels=self.labels1, embeddings=self.o1, margin=self.margin_in_loss, squared=True)


    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):
            if self.res_n < 50 :
                residual_block = ops_resnet.resblock
            else :
                residual_block = ops_resnet.bottle_resblock

            residual_list = ops_resnet.get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = ops_resnet.conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################

            x = ops_resnet.batch_norm(x, is_training, scope='batch_norm')
            x = ops_resnet.relu(x)

            x = ops_resnet.global_avg_pooling(x)
            x = ops_resnet.fully_conneted(x, units=self.feature_space_dimension, scope='logit')

            return x


    def batch_hard_triplet_loss(self, labels, embeddings, margin, squared=False):  #--> Furthest_Nearest_batch_triplet_loss
        # https://github.com/omoindrot/tensorflow-triplet-loss
        # https://omoindrot.github.io/triplet-loss
        # https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = Utils_losses.pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = Utils_losses.get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

        # shape (batch_size, 1)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = Utils_losses.get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_pairwise_dist_rowwise = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_pairwise_dist_rowwise * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss
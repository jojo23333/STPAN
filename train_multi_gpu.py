import os, time, argparse
import random, sys, cv2, os
import tensorflow as tf
import numpy as np
import pandas as pd

from datetime import datetime
from PIL import Image

from train_tools import sRGBforward, img_loss, average_gradients
from data_loader import DataLoader
from stpan import stpan

FLAGS = tf.flags.FLAGS

# model settings
tf.flags.DEFINE_integer('frame_size', 5, 'Input number of frame')
tf.flags.DEFINE_float('degamma', 2.2, 'Degamma parameter for inverse gamma correction')
tf.flags.DEFINE_integer('input_size', 128, 'Height and width for input image patches for train')
tf.flags.DEFINE_integer('max_offset', 128, 'Default offset is obtained by multiply max_offset after normlized.')
tf.flags.DEFINE_string('noise_level_type', 'single_std', 'Noise level type input into the network')
tf.flags.DEFINE_integer('kernel_size_t', 3, 'Default kernel size on the temporal dimension. T*K*K')
tf.flags.DEFINE_integer('kernel_dilate_rate', 1, 'Set this term > 1 if you want to use dilated kernel.')
tf.flags.DEFINE_integer('kernel_size_k', 3 , 'Default kernel size on the spatial dimension. T*K*K')
tf.flags.DEFINE_bool('color', False, 'Whether to denoise on colored image. DEPRECATED')

# Training settings
tf.flags.DEFINE_integer('batch_size', 8, 'Number of images in a single batch')
tf.flags.DEFINE_float('base_lr', 2e-4, 'Start learning rate')
tf.flags.DEFINE_float('bottom_lr', 1e-4, 'Bottom learning rate')
tf.flags.DEFINE_float('decay_rate', 0.995, 'Decay rate')
tf.flags.DEFINE_float('anneal_alpha', .9997, 'Annealing rate')
tf.flags.DEFINE_bool('use_anneal_loss', True, 'Whether to use anneal loss')
tf.flags.DEFINE_integer('max_epoch', 300, 'Max training epoch')
tf.flags.DEFINE_string('train_csv', '../train_data_Scene_5.csv', 'csv file for training input. DEPRECATED.')
tf.flags.DEFINE_string('run_root', '/data/3DF/', 'Default output dir of checkpoint and logs.')
tf.flags.DEFINE_string('norm_type', 'none', "use 'batch' to indicate batch normalization and 'group' to indicate group normalization. DEPRECATED.")
tf.flags.DEFINE_integer('gpu', 1, 'Number of gpu used for training')
tf.flags.DEFINE_string('dataset', '', 'dataset path')
tf.flags.DEFINE_string('model_path', '', 'Input model path')
tf.flags.DEFINE_string('name', '', 'Saved run name')


class TrainSolverMultiGPU(object):
    """
        Train wrapper for multiple GPU training
    """
    def __init__(self):
        """
            Some setting up for the output dirs
        """
        if not os.path.exists("runs"):
            os.mkdir("runs")
        time = datetime.strftime(datetime.now(), "%Y-%m-%d-%H_%M")
        if FLAGS.name:
            self.run_root = os.path.join("./runs", FLAGS.name)
        else:
            self.run_root = os.path.join("./runs", time)
        self.log_root = os.path.join(self.run_root, "log")
        self.checkpoint_root = os.path.join(self.run_root, "checkpoint")
        if not os.path.exists(self.run_root):
            os.mkdir(self.run_root)
        if not os.path.exists(self.log_root):
            os.mkdir(self.log_root)
        if not os.path.exists(self.checkpoint_root):
            os.mkdir(self.checkpoint_root)

    def build_parallel(self, gs, data_iterator, dataset_size):
        """
            Build model for training, variables saved at 
        """
        BATCH_SIZE = FLAGS.batch_size
        INPUT_H = INPUT_W = FLAGS.input_size
        FRAME_SIZE = FLAGS.frame_size
        COLOR = FLAGS.color
        COLOR_CH = 3 if COLOR else 1

        BASE_LR = FLAGS.base_lr
        BOTTOM_LR = FLAGS.bottom_lr
        DECAY_RATE = FLAGS.decay_rate
        ANNEAL_ALPHA = FLAGS.anneal_alpha
        USE_ANNEAL_LOSS = FLAGS.use_anneal_loss
        NOISE_LEVEL_TYPE = FLAGS.noise_level_type
        NUM_GPU = FLAGS.gpu
        MINI_BATCH_SIZE = BATCH_SIZE
        NORM_TYPE = FLAGS.norm_type
        KERNEL_T = FLAGS.kernel_size_t

        anneal_term = tf.pow(ANNEAL_ALPHA, tf.cast(gs, tf.float32)) * 100.0

        # Leaning rate & Optimizer
        DECAY_STEP_SIZE = dataset_size // (BATCH_SIZE * NUM_GPU)
        lr = tf.train.exponential_decay(BASE_LR, gs, DECAY_STEP_SIZE ,DECAY_RATE, staircase=True)
        lr = tf.maximum(lr, BOTTOM_LR)
        optimizer = tf.train.AdamOptimizer(lr)

        model = df3c_only_v2

        train_input = tf.placeholder(dtype='float32', shape=[MINI_BATCH_SIZE, INPUT_H, INPUT_W, FRAME_SIZE, 1])
        noise_level = tf.placeholder(dtype='float32', shape=[MINI_BATCH_SIZE, INPUT_H, INPUT_W, 1])
        with tf.device('/CPU:0'):
            with tf.name_scope('cpu'):
                with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                    model(train_input, noise_level, NOISE_LEVEL_TYPE, anneal_term=1, 
                            C=COLOR_CH, norm_type=NORM_TYPE, batch_size=MINI_BATCH_SIZE)
      
        tower_grads = []
        base_loss_all = []
        ref_loss_all = []
        w_mean_all = []
        t_mean_all = []
        # build up parallel pipeline
        for i in range(NUM_GPU):
            with tf.device('/GPU:%d' % i):
                with tf.name_scope('gpu_%d' % i):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        next_train_element = data_iterator.get_next()
                        train_input, train_gt, img_name, noise_level, wl = next_train_element
                        sig_read_single_std = tf.sqrt(noise_level[...,0:1] ** 2 + tf.maximum(0., train_input[...,FRAME_SIZE//2,:]) * noise_level[...,1:2] ** 2)
                        noise_level = noise_level if NOISE_LEVEL_TYPE == "double" else sig_read_single_std if NOISE_LEVEL_TYPE == "single_std" else None

                        x, op_flow, w, per_point_out = model(train_input, noise_level, NOISE_LEVEL_TYPE, anneal_term=1, 
                                                            C=COLOR_CH, norm_type=NORM_TYPE, batch_size=MINI_BATCH_SIZE)

                        base_loss = img_loss(sRGBforward(x / wl), sRGBforward(train_gt / wl))

                        part_points_loss = 0
                        ref_loss =[]
                        for i in range(KERNEL_T):
                            part_out = tf.reduce_sum(per_point_out[...,i*9:(i+1)*9,:], axis=-2) * KERNEL_T
                            ref_loss.append(img_loss(sRGBforward(part_out / wl), sRGBforward(train_gt / wl)))
                            part_points_loss += img_loss(sRGBforward(part_out / wl), sRGBforward(train_gt / wl))
                        loss = base_loss
                        if USE_ANNEAL_LOSS:
                            loss += part_points_loss * anneal_term
                        t_mean = tf.reduce_mean(tf.abs(op_flow[...,2]))
                        w_mean = tf.reduce_mean(tf.abs(w))
                        # save frads
                        grads = optimizer.compute_gradients(loss)
                        tower_grads.append(grads)
                        # save loss information
                        base_loss_all.append(base_loss)
                        ref_loss_all.append(ref_loss)
                        w_mean_all.append(w_mean)
                        t_mean_all.append(t_mean)
        
        grads = average_gradients(tower_grads)
        apply_grads_op = optimizer.apply_gradients(grads, global_step=gs)

        # count averaged information
        base_loss = sum(base_loss_all) / NUM_GPU
        w_mean = sum(w_mean_all) / NUM_GPU
        t_mean = sum(t_mean_all) / NUM_GPU
        ref_loss = [ sum([ref_loss_all[j][i] for j in range(NUM_GPU)]) / NUM_GPU for i in range(KERNEL_T)]

        return (base_loss, ref_loss, gs, lr, apply_grads_op, anneal_term, w_mean, t_mean)


    def train(self):
        """
            train the network
        """
        MODEL_PATH = FLAGS.model_path
        FRAME_SIZE = FLAGS.frame_size
        INPUT_H = INPUT_W = FLAGS.input_size
        BATCH_SIZE = FLAGS.batch_size
        MAX_EPOCH = FLAGS.max_epoch
        COLOR = FLAGS.color
        DEGAMMA = FLAGS.degamma

        # Dataset for train and validation
        data_loader = DataLoader()
        with tf.device('/CPU:0'):
            train_dataset, val_dataset = data_loader.build_train_pipeline(batch_size=BATCH_SIZE)
            data_iterator = tf.data.Iterator.from_structure(val_dataset.output_types, val_dataset.output_shapes)
            train_init_op = data_iterator.make_initializer(train_dataset)
            val_init_op = data_iterator.make_initializer(val_dataset)

        # Global step & Anneal terms
        gs = tf.Variable(0, trainable=False)

        # use locals().update() to unpack local tensor in build fuction to this fuction
        tf_locals = self.build_parallel(gs, data_iterator, data_loader.dataset_size)
        base_loss, ref_loss, gs, lr, apply_grads_op, anneal_term, w_mean, t_mean = tf_locals

        # Set up for summary
        saver = tf.train.Saver(max_to_keep=10)
        # tf.summary.scalar("total loss", loss)
        tf.summary.scalar("base loss", base_loss)
        tf.summary.scalar("annealed term", anneal_term)
        tf.summary.scalar("learning rate", lr)
        tf.summary.scalar("w_mean", w_mean)
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            # Tensorboard open log with "tensorbard --logdir=logs"
            file_writer = tf.summary.FileWriter(self.log_root, sess.graph)

            sess.run(tf.global_variables_initializer())
            if MODEL_PATH:
                print("Restoring model from path:", MODEL_PATH)
                saver.restore(sess, MODEL_PATH)
                print("Done")

            self.val_loss = 1
            # print infomation when OMM happens
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
            for epoch in range(0, MAX_EPOCH):
                # reinitialize the data_iterator, each time iterate for a whole batch before stop
                sess.run(train_init_op)
                cnt = .0
                base_loss_sum = .0
                ref_loss_sum = .0
                while True:
                    try:
                        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        # run_metadata = tf.RunMetadata()
                        start_time = time.time()
                        _, bl, rl, g_step, summary, t_mean_ = sess.run([apply_grads_op, base_loss, ref_loss,  gs, merged, t_mean]
                            , options=run_options)#, run_metadata=run_metadata)
                        duration = time.time() - start_time
                        if duration > 20:
                            print("DATASET LOADING DURATION: %.3f" % duration)
                        if not g_step % 400:
                            print("[epoch %d gloabal_step %d] base_loss %.6f ref_loss" % (epoch, g_step, bl), rl,
                                    "t_mean %f time_per_batch: %.3f" % (t_mean_, duration))
                        if g_step % 10 == 0:
                            # file_writer.add_run_metadata(run_meta_data, 'step%d' % g_step)
                            file_writer.add_summary(summary, g_step)
                        sys.stdout.flush()
                        base_loss_sum += bl
                        ref_loss_sum += sum(rl) #if sttn_only else rl
                        cnt += 1
                    except tf.errors.OutOfRangeError:
                        break
                lr_ , at = sess.run([lr, anneal_term])
                print("learning_rate: %.5f \t annealed_term: %.4f\n \t base_loss: %.6f\t ref_loss: %.6f"\
                        %  (lr_, at, base_loss_sum / cnt, ref_loss_sum / cnt))

                # save session after each iteration through dataset
                if epoch % 2 == 0:
                    cnt = .0
                    base_loss_sum = .0
                    sess.run(val_init_op)
                    while True:
                        try:
                            start_time = time.time()
                            bl = sess.run(base_loss)
                            duration = time.time() - start_time
                            if duration > 20:
                                print("validation LOADING DURATION: %.3f" % duration)
                            base_loss_sum += bl
                            cnt += 1
                            print("Validation loss %f on %d" % (bl, cnt), end='\r')
                        except tf.errors.OutOfRangeError:
                            break
                    val_loss = base_loss_sum / cnt
                    if val_loss < self.val_loss:
                        save_path = saver.save(sess, os.path.join(self.checkpoint_root, "df3c-%d.ckpt" % (g_step // 1000)))
                        print("Validation loss imporve from %f to %f" % (self.val_loss, val_loss))
                        print("Model saved at ", save_path)
                        self.val_loss = val_loss
                    else:
                        print("Validation loss:%f do not imporve from %f" % (val_loss, self.val_loss))


def main(argv):
    solver = TrainSolverMultiGPU()
    solver.train()

if __name__ == "__main__":
    tf.app.run()

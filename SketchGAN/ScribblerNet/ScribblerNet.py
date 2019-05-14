from operators import *
from utils import *
from random import shuffle
import re

import tensorflow as tf
import scipy.io as sio

class Model(object):
    def __init__(self, sess, train_root = "./DataSet", test_root = "./DataSet", rst_root = "./DataSet", batch_size = 4):
        self.sess = sess
        self.train_root = train_root
        self.test_root = test_root
        self.rst_root = rst_root
        self.batch_size = batch_size
        self.img_height = 256
        self.img_width = 256
        self.learning_rate = 0.0001
        self.lam = 10
        self.gen_learning_rate = 0.0001
        self.dis_learning_rate = 0.0003
        self.critic_iteration = 1

        self.is_train = True
        self.buildModel()

    def save(self, checkpoint_dir, step):
        model_name = "ScibblerNet.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def GeneratorNet(self, img, reuse=False):
        with tf.variable_scope("gen") as scope:
            if reuse:
                scope.reuse_variables()
            
            # 256 x 256 x 1 --> 256 x 256 x 32
            layer_0 = genConv2d(img, 32, name="gen_0_conv")
            # 256 x 256 x 32 --> 256 x 256 x 32
            layer_1 = genConv2d(layer_0, 32, name="gen_1_conv")
            # 256 x 256 x 32 --> 128 x 128 x 64
            layer_2 = genConv2d(layer_1, 64, strides=[2, 2], name="gen_2_conv")
            # 128 x 128 x 64 --> 128 x 128 x 64
            layer_3 = genConv2d(layer_2, 64, name="gen_3_conv")
            # 128 x 128 x 64 --> 64 x 64 x 128
            layer_4 = genConv2d(layer_3, 128, strides=[2, 2], name="gen_4_conv")
            # 64 x 64 x 128 --> 64 x 64 x 128
            layer_5 = genConv2d(layer_4, 128, name="gen_5_conv")
            # 64 x 64 x 128 --> 32 x 32 x 256
            layer_6 = genConv2d(layer_5, 256, strides=[2, 2], name="gen_6_conv")
            # 32 x 32 x 256 --> 32 x 32 x 256
            layer_7 = genConv2d(layer_6, 256, name="gen_7_conv")
            layer_7 = tf.nn.relu(layer_6 + layer_7)
            # 32 x 32 x 256 --> 32 x 32 x 256
            layer_8 = genConv2d(layer_7, 256, name="gen_8_conv")
            layer_8 = tf.nn.relu(layer_7 + layer_8)
            # 32 x 32 x 256 --> 32 x 32 x 256
            layer_9 = genConv2d(layer_8, 256, name="gen_9_conv")
            layer_9 = tf.nn.relu(layer_8 + layer_9)
            # 32 x 32 x 256 --> 32 x 32 x 256
            layer_10 = genConv2d(layer_9, 256, name="gen_10_conv")
            layer_10 = tf.nn.relu(layer_9 + layer_10)
            # 32 x 32 x 256 --> 32 x 32 x 256
            layer_11 = genConv2d(layer_10, 256, name="gen_11_conv")
            layer_11 = tf.nn.relu(layer_10 + layer_11)
            # 32 x 32 x 256 --> 32 x 32 x 256
            layer_12 = genConv2d(layer_11, 256, name="gen_12_conv")
            layer_12 = tf.nn.relu(layer_11 + layer_12)
            # 32 x 32 x 256 --> 32 x 32 x 128
            layer_13 = genConv2d(layer_12, 128, name="gen_13_conv")
            # 32 x 32 x 128 --> 64 x 64 x 128
            layer_14 = genDeconv2d(layer_13, 128, name="gen_14_deconv")
            # 64 x 64 x 128 --> 64 x 64 x 128
            layer_15 = genConv2d(layer_14, 128, name="gen_15_conv")
            # 64 x 64 x 128 --> 64 x 64 x 128
            layer_16 = genConv2d(layer_15, 128, name="gen_16_conv")
            # 64 x 64 x 64 --> 64 x 64 x 64
            layer_17 = genConv2d(layer_16, 64, name="gen_17_conv")
            # 64 x 64 x 64 --> 128 x 128 x 64
            layer_18 = genDeconv2d(layer_17, 64, name="gen_18_deconv")
            # 128 x 128 x 64 --> 128 x 128 x 64
            layer_19 = genConv2d(layer_18, 64, name="gen_19_conv")
            # 128 x 128 x 64 --> 128 x 128 x 64
            layer_20 = genConv2d(layer_19, 64, name="gen_20_conv")
            # 128 x 128 x 64 --> 128 x 128 x 32
            layer_21 = genConv2d(layer_20, 32, name="gen_21_conv")
            # 128 x 128 x 32 --> 256 x 256 x 32
            layer_22 = genDeconv2d(layer_21, 32, name="gen_22_deconv")
            # 256 x 256 x 32 --> 256 x 256 x 32
            layer_23 = genConv2d(layer_22, 32, name="gen_23_conv")
            # 256 x 256 x 32 --> 256 x 256 x 2
            layer_24 = genConv2d(layer_23, 4, name="gen_24_conv")
            
            return layer_24

    def DiscriminatorNet(self, img, ori, reuse=False):
        with tf.variable_scope("dis") as scope:
            if reuse:
                scope.reuse_variables()

            # 256 x 256 x 4 concat 256 x 256 x 4
            layer_0 = cropConcat(img, ori, 3, name="dis_0_concat")
            # 256 x 256 x 4 --> 128 x 128 x 32
            layer_1 = disConv2d(layer_0, 32, strides=[2, 2], name="dis_1_conv")
            # 128 x 128 x 32 --> 64 x 64 x 64
            layer_2 = disConv2d(layer_1, 64, strides=[2, 2], name="dis_2_conv")
            # 64 x 64 x 64 --> 32 x 32 x 256
            layer_3 = disConv2d(layer_2, 256, strides=[2, 2], name="dis_3_conv")
            # 32 x 32 x 256 --> 32 x 32 x 256
            layer_4 = disConv2d(layer_3, 256, name="dis_4_conv")
            layer_4 = layer_3 + layer_4
            # 32 x 32 x 256 --> 32 x 32 x 256
            layer_5 = disConv2d(layer_4, 256, name="dis_5_conv")
            layer_5 = layer_4 + layer_5
            # 32 x 32 x 256 --> 16 x 16 x 128
            layer_6 = disConv2d(layer_5, 128, strides=[2, 2], name="dis_6_conv")
            

            d = tf.reshape(layer_6, [self.batch_size, -1])
            d = linear(d, 1, 'dis_6_lin')
            return d, layer_5, layer_3, layer_2, layer_1

    def contentLayerLoss(self, p, x):
        b, h, w, c = p.get_shape()
        k = 1./(2.*b.value)
        loss = k * tf.reduce_sum(tf.pow(p - x, 2))
        return loss

    def gramMatrix(self, x, batch_size, area, depth):
        F = tf.reshape(x, [batch_size, area, depth])
        TF = tf.transpose(F, perm=[0, 2, 1])
        G = tf.matmul(TF, F)
        return G
    
    def styleLayerLoss(self, a, x):
        b, h, w, c = a.get_shape()
        M = h.value * w.value
        N = c.value
        A = self.gramMatrix(a, b.value, M, N)
        X = self.gramMatrix(x, b.value, M, N)
        loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((X - A), 2))
        return loss / b.value

    def buildModel(self):
        input_dims = [self.img_height, self.img_width, 1]
        gt_dims = [self.img_height, self.img_width, 4]

        self.img_input = tf.placeholder(tf.float32, [self.batch_size] + input_dims, name="img_input")
        self.ori_gt = tf.placeholder(tf.float32, [self.batch_size] + gt_dims, name="ori_gt")

        # generator output
        self.ori_output = self.GeneratorNet(self.img_input, reuse=False)

        # optimize energy
        self.D_real, dr_l3, dr_l2, dr_l1, dr_l0 = self.DiscriminatorNet(self.img_input, self.ori_gt, reuse=False)
        self.D_fake, df_l3, df_l2, df_l1, df_l0 = self.DiscriminatorNet(self.img_input, self.ori_output, reuse=True)
        
        # Discriminator loss
        eps = tf.random_uniform([self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        x_inter = eps * self.ori_gt + (1. - eps) * self.ori_output
        D_x_inter, _, _, _, _ = self.DiscriminatorNet(self.img_input, x_inter, reuse=True)
        grad_v = tf.gradients(D_x_inter, [x_inter])[0]
        grad_v = tf.reshape(grad_v, [self.batch_size, -1])
        grad_norm = tf.reduce_sum(grad_v ** 2, axis=1)
        grad_norm = tf.sqrt(grad_norm)
        grad_pen = self.lam * tf.reduce_mean((grad_norm - 1.) ** 2)
        self.D_real_mean = tf.reduce_mean(self.D_real)
        self.D_fake_mean = tf.reduce_mean(self.D_fake)
        self.D_loss = tf.reduce_mean(self.D_fake) - tf.reduce_mean(self.D_real) + grad_pen

        # Generator loss
        # self.data_loss = tf.losses.absolute_difference(self.ori_gt, self.ori_output)
        '''
        self.style_loss = self.styleLayerLoss(df_l3, dr_l3) + self.styleLayerLoss(df_l2, dr_l2) \
                            + self.styleLayerLoss(df_l1, dr_l1) + self.styleLayerLoss(df_l0, dr_l0) \
                            + self.styleLayerLoss(self.ori_output, self.ori_gt)
        '''
        self.content_loss = 0.5 * self.contentLayerLoss(self.ori_output, self.ori_gt) + 0.5 * self.contentLayerLoss(df_l2, dr_l2)
        
        # for content loss
        k_weight_alpha = 1.e-2
        # for style loss
        # k_weight_beta = 1.e-2 * k_weight_alpha

        # self.G_loss = k_weight_alpha * self.content_loss + k_weight_beta * self.style_loss
        self.G_loss = k_weight_alpha * self.content_loss

        t_vars = tf.trainable_variables()
        self.G_vars = [var for var in t_vars if "gen_" in var.name]
        self.D_vars = [var for var in t_vars if "dis_" in var.name]
        self.saver = tf.train.Saver()

    def getShufferTrainBatch(self, idx, train_list, epoch_size):
        if idx % epoch_size == 0:
            shuffle(train_list)
        offset = (idx % epoch_size) * self.batch_size
        train_img_data, train_ori_gt = getTrainBatch(train_list, offset, self.batch_size)
        return train_img_data, train_ori_gt, idx + 1
    
    def train(self, max_iter):
        D_solver = tf.train.AdamOptimizer(learning_rate=self.dis_learning_rate, beta1=0., beta2=0.9)\
            .minimize(self.D_loss, var_list=self.D_vars)
        G_solver = tf.train.AdamOptimizer(learning_rate=self.gen_learning_rate, beta1=0., beta2=0.9)\
            .minimize(self.G_loss, var_list=self.G_vars)
        tf.global_variables_initializer().run()

        test_list = getFileSetList(self.test_root)
        test_input, test_gt = getTrainBatch(test_list, 240, self.batch_size)
        normal_gt_show, depth_gt_show = draw2DOri(test_gt, self.batch_size)
        cv2.imwrite(self.rst_root + "/normal_GT.png", normal_gt_show)
        cv2.imwrite(self.rst_root + "/depth_GT.png", depth_gt_show)

        for i_test in range(self.batch_size):
            cv2.imwrite(self.rst_root + "/" + str(i_test) + "_channel_0.png", test_input[i_test, :, :, 0] * 255.)

        train_list = getFileSetList(self.train_root)
        epoch_size = len(train_list) // self.batch_size
        load_flag, ckpt_counter = self.load(self.rst_root + "/checkpoint")
        begin_idx = ckpt_counter + 1 if load_flag else 0

        idx = 0
        self.save_D_loss = []
        self.save_G_loss = []
        self.save_test_D_loss = []
        self.save_test_D_count = []
        for i_iter in range(begin_idx, max_iter):
            for c in range(self.critic_iteration):
                batch_img, batch_gt, idx = self.getShufferTrainBatch(idx, train_list, epoch_size)
                _, D_loss, D_real, D_fake = self.sess.run([D_solver, self.D_loss, self.D_real_mean, self.D_fake_mean],
                                    feed_dict={self.img_input: batch_img, self.ori_gt: batch_gt})
                print("(%d, %d): D_loss: %.8f, D_real: %.8f, D_fake: %.8f" % (i_iter, c, D_loss, D_real, D_fake))

            batch_img, batch_gt, idx = self.getShufferTrainBatch(idx, train_list, epoch_size)
            '''
            _, G_loss, D_loss, content_loss, style_loss =  self.sess.run([G_solver, self.G_loss, self.D_loss,
                                                                        self.content_loss, self.style_loss],
                                                                        feed_dict={self.img_input: batch_img, self.ori_gt: batch_gt})
            print("Iter %d: D_loss: %.8f, G_loss: %.8f, content_loss: %.8f, style_loss: %.8f" % 
                (i_iter, D_loss, G_loss, content_loss, style_loss))
            '''
            _, G_loss, D_loss, content_loss =  self.sess.run([G_solver, self.G_loss, self.D_loss, self.content_loss],
                                                                        feed_dict={self.img_input: batch_img, self.ori_gt: batch_gt})
            print("Iter %d: D_loss: %.8f, G_loss: %.8f, content_loss: %.8f" % 
                (i_iter, D_loss, G_loss, content_loss))

            self.save_D_loss.append(-D_loss)
            self.save_G_loss.append(G_loss)

            if(i_iter % 100 == 0 or i_iter == max_iter - 1):
                self.is_train = False
                test_ori, D_loss, G_loss = self.sess.run([self.ori_output, self.D_loss, self.G_loss],
                                            feed_dict={self.img_input: test_input, self.ori_gt: test_gt})

                self.save_test_D_loss.append(-D_loss)
                self.save_test_D_count.append(i_iter)

                test_ori = np.array(test_ori)
                normal_result_show, depth_result_show = draw2DOri(test_ori, self.batch_size)
                cv2.imwrite(self.rst_root + "/t_%d_depth.png" % i_iter, depth_result_show)
                cv2.imwrite(self.rst_root + "/t_%d_normal.png" % i_iter, normal_result_show)

                self.save(self.rst_root + "/checkpoint/", i_iter)

                D_loss_array = np.array(self.save_D_loss)
                G_loss_array = np.array(self.save_G_loss)
                test_D_loss_array = np.array(self.save_test_D_loss)
                test_D_count_array = np.array(self.save_test_D_count)
                if not os.path.exists(self.rst_root + "/Loss/"):
                    os.makedirs(self.rst_root + "/Loss/")
                sio.savemat(self.rst_root + "/Loss/loss_%d.mat" % i_iter, {"d_loss": D_loss_array, "g_loss": G_loss_array,
                                                                        "test_d_loss": test_D_loss_array, "test_d_count": test_D_count_array})

                self.is_train = True
                self.save_D_loss.clear()
                self.save_G_loss.clear()
                self.save_test_D_count.clear()
                self.save_test_D_loss.clear()

    def test(self, input_dir):
        load_flag, ckpt_counter = self.load(self.rst_root + "/checkpoint/")
        if not load_flag:
            print("No parameters!")
            return
        import time
        self.is_train = False
        since_time = time.time()
        input_list = getFileSetList(input_dir)
        input_data = loadImgInput(input_list)
        input_data = np.resize(input_data, [1, self.img_height, self.img_width, 1])
        
        test_input = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, 1])
        test_ori = self.GeneratorNet(test_input, reuse=True)
        test_result = self.sess.run([test_ori], feed_dict={test_input: input_data})
        normal_result_show, depth_result_show = draw2DOri(test_result, 1)
        depth_result_show = np.expand_dims(depth_result_show, axis=2)
        print("depth reuslt's shape", depth_result_show.shape)
        depth_result_show_r = depth_result_show
        depth_result_show_g = np.concatenate((depth_result_show, depth_result_show_r), axis=2)
        depth_result_show = np.concatenate((depth_result_show_g, depth_result_show), axis=2)
        print("depth reuslt's new shape", depth_result_show.shape)
        cv2.imwrite(input_dir + "/t_depth.jpg", depth_result_show)
        cv2.imwrite(input_dir + "/t_normal.jpg", normal_result_show)
        print("--- %s seconds ---" % (time.time() - since_time))

    def test_sequence(self, input_dir):
        self.is_train = False
        since_time = time.time()
        input_list = getFileSetList(input_dir)
        input_data = loadImgInput(input_list)
        input_data = np.resize(input_data, [1, self.img_height, self.img_width, 1])

        test_input = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, 1])
        test_ori = self.GeneratorNet(test_input, reuse=True)
        test_result = self.sess.run([test_ori], feed_dict={test_input: input_data})
        normal_result_show, depth_result_show = draw2DOri(test_result, 1)
        depth_result_show = np.expand_dims(depth_result_show, axis=2)
        print("depth reuslt's shape", depth_result_show.shape)
        depth_result_show_r = depth_result_show
        depth_result_show_g = np.concatenate((depth_result_show, depth_result_show_r), axis=2)
        depth_result_show = np.concatenate((depth_result_show_g, depth_result_show), axis=2)
        print("depth reuslt's new shape", depth_result_show.shape)
        cv2.imwrite(input_dir + "/t_depth.jpg", depth_result_show)
        cv2.imwrite(input_dir + "/t_normal.jpg", normal_result_show)
        print("--- %s seconds ---" % (time.time() - since_time))
        

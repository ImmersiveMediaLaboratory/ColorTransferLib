"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.

Adaptation of https://github.com/cysmith/neural-style-tf
"""

import tensorflow as tf
import numpy as np
import torch
import time
import cv2
import os
from copy import deepcopy

from ColorTransferLib.Utils.Helper import check_compatibility
from ColorTransferLib.Algorithms.NST.Model import Model


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: A Neural Algorithm of Artistic Style
#   Author: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
#   Published in: arXiv
#   Year of Publication: 2015
#
# Abstract:
#    In fine art, especially painting, humans have mastered the skill to create unique visual experiences through
#    composing a complex interplay between the content and style of an image. Thus far the algorithmic basis of this
#    process is unknown and there exists no artificial system with similar capabilities. However, in other key areas of
#    visual perception such as object and face recognition near-human performance was recently demonstrated by a class
#    of biologically inspired vision models called Deep Neural Networks. Here we introduce an artificial system based on
#    a Deep Neural Network that creates artistic images of high perceptual quality. The system uses neural
#    representations to separate and recombine content and style of arbitrary images, providing a neural algorithm for
#    the creation of artistic images. Moreover, in light of the striking similarities between performance-optimised
#    artificial neural networks and biological vision, our work offers a path forward to an algorithmic understanding of
#    how humans create and perceive artistic imagery.
#
# Info:
#   Name: NeuralStyleTransfer
#   Identifier: NST
#   Link: https://doi.org/10.48550/arXiv.1508.06576
#   Original Source Code: https://github.com/cysmith/neural-style-tf
#
# Misc:
#   Note: the neural_style.py is adapted to support TensorFLow 2
#
# Implementation Details:
#   optimizer = ADAM, mbecause L-BFGS not working on TF2
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class NST:
    compatibility = {
        "src": ["Image", "Mesh"],
        "ref": ["Image", "Mesh"]
    }

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "NST",
            "title": "A Neural Algorithm of Artistic Style",
            "year": 2015,
            "abstract": "In fine art, especially painting, humans have mastered the skill to create unique visual "
                        "experiences through composing a complex interplay between the content and style of an image. "
                        "Thus far the algorithmic basis of this process is unknown and there exists no artificial "
                        "system with similar capabilities. However, in other key areas ofvisual perception such as "
                        "object and face recognition near-human performance was recently demonstrated by a class "
                        "of biologically inspired vision models called Deep Neural Networks. Here we introduce an "
                        "artificial system based on a Deep Neural Network that creates artistic images of high "
                        "perceptual quality. The system uses neural representations to separate and recombine content "
                        "and style of arbitrary images, providing a neural algorithm for the creation of artistic "
                        "images. Moreover, in light of the striking similarities between performance-optimised "
                        "artificial neural networks and biological vision, our work offers a path forward to an "
                        "algorithmic understanding of how humans create and perceive artistic imagery.",
            "types": ["Image"]
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, NST.compatibility)

        if output["status_code"] == -1:
            output["response"] = "Incompatible type."
            return output
        
        if not torch.cuda.is_available():
            opt.device = "/cpu:0"

        # Preprocessing
        src_img = src.get_raw() * 255.0
        ref_img = ref.get_raw() * 255.0
        out_img = deepcopy(src)

        opt.style_layer_weights = NST.normalize(opt.style_layer_weights)
        opt.content_layer_weights = NST.normalize(opt.content_layer_weights)
        opt.style_imgs_weights = NST.normalize(opt.style_imgs_weights)

        h, w, d = src_img.shape
        mx = opt.max_size
        # resize if > max size
        if h > w and h > mx:
            w = (float(mx) / float(h)) * w
            src_img = cv2.resize(src_img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
        if w > mx:
            h = (float(mx) / float(w)) * h
            src_img = cv2.resize(src_img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
        src_img = NST.preprocess(src_img)

        _, ch, cw, cd = src_img.shape
        style_imgs = []
        ref_img = cv2.resize(ref_img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
        ref_img = NST.preprocess(ref_img)
        style_imgs.append(ref_img)

        out = NST.render_single_image(src_img, style_imgs, opt)
        out = NST.postprocess(out)
        tf.compat.v1.reset_default_graph()

        out_img.set_raw(out, normalized=True)
        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output

    # ------------------------------------------------------------------------------------------------------------------
    # 'a neural algorithm for artistic style' loss functions
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def content_layer_loss(p, x, opt):
        _, h, w, d = p.get_shape()
        M = h * w
        N = d
        if opt.content_loss_function == 1:
            K = 1. / (2. * N ** 0.5 * M ** 0.5)
        elif opt.content_loss_function == 2:
            K = 1. / (N * M)
        elif opt.content_loss_function == 3:
            K = 1. / 2.
        loss = K * tf.reduce_sum(input_tensor=tf.pow((x - p), 2))
        return loss

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def style_layer_loss(a, x):
        _, h, w, d = a.get_shape()
        M = h * w
        N = d
        A = NST.gram_matrix(a, M, N)
        G = NST.gram_matrix(x, M, N)
        loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(input_tensor=tf.pow((G - A), 2))
        return loss

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def gram_matrix(x, area, depth):
        F = tf.reshape(x, (area, depth))
        G = tf.matmul(tf.transpose(a=F), F)
        return G

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mask_style_layer(a, x, mask_img, opt):
        _, h, w, d = a.get_shape()
        mask = NST.get_mask_image(mask_img, w, h, opt)
        mask = tf.convert_to_tensor(value=mask)
        tensors = []
        for _ in range(d):
            tensors.append(mask)
        mask = tf.stack(tensors, axis=2)
        mask = tf.stack(mask, axis=0)
        mask = tf.expand_dims(mask, 0)
        a = tf.multiply(a, mask)
        x = tf.multiply(x, mask)
        return a, x

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def sum_masked_style_losses(sess, net, style_imgs, opt):
        total_style_loss = 0.
        weights = opt.style_imgs_weights
        masks = opt.style_mask_imgs

        for img, img_weight, img_mask in zip(style_imgs, weights, masks):
            sess.run(net['input'].assign(img))
            style_loss = 0.
            for layer, weight in zip(opt.style_layers, opt.style_layer_weights):
                a = sess.run(net[layer])
                x = net[layer]
                a = tf.convert_to_tensor(value=a)
                a, x = NST.mask_style_layer(a, x, img_mask, opt)
                style_loss += NST.style_layer_loss(a, x) * weight
            style_loss /= float(len(opt.style_layers))
            total_style_loss += (style_loss * img_weight)
        total_style_loss /= float(len(style_imgs))
        return total_style_loss

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def sum_style_losses(sess, net, style_imgs, opt):
        total_style_loss = 0.
        weights = opt.style_imgs_weights
        for img, img_weight in zip(style_imgs, weights):
            sess.run(net['input'].assign(img))
            style_loss = 0.
            for layer, weight in zip(opt.style_layers, opt.style_layer_weights):
                a = sess.run(net[layer])
                x = net[layer]
                a = tf.convert_to_tensor(value=a)
                style_loss += NST.style_layer_loss(a, x) * weight
            style_loss /= float(len(opt.style_layers))
            total_style_loss += (style_loss * img_weight)
        total_style_loss /= float(len(style_imgs))
        return total_style_loss

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def sum_content_losses(sess, net, content_img, opt):
        sess.run(net['input'].assign(content_img))
        content_loss = 0.
        for layer, weight in zip(opt.content_layers, opt.content_layer_weights):
            p = sess.run(net[layer])
            x = net[layer]
            p = tf.convert_to_tensor(value=p)
            content_loss += NST.content_layer_loss(p, x, opt) * weight
        content_loss /= float(len(opt.content_layers))
        return content_loss

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def preprocess(img):
        imgpre = np.copy(img)
        # bgr to rgb
        imgpre = imgpre[..., ::-1]
        # shape (h, w, d) to (1, h, w, d)
        imgpre = imgpre[np.newaxis, :, :, :]
        imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
        return imgpre

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def postprocess(img):
        imgpost = np.copy(img)
        imgpost += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
        # shape (1, h, w, d) to (h, w, d)
        imgpost = imgpost[0]
        imgpost = np.clip(imgpost, 0, 255).astype('uint8')
        # rgb to bgr
        imgpost = imgpost[..., ::-1]
        return imgpost

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def normalize(weights):
        denom = sum(weights)
        if denom > 0.:
            return [float(i) / denom for i in weights]
        else:
            return [0.] * len(weights)

    # ------------------------------------------------------------------------------------------------------------------
    # rendering -- where the magic happens
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def stylize(content_img, style_imgs, init_img, opt, frame=None):
        with tf.device(opt.device), tf.compat.v1.Session() as sess:
            # setup network
            net = Model(content_img, opt).net

            # style loss
            if opt.style_mask:
                L_style = NST.sum_masked_style_losses(sess, net, style_imgs, opt)
            else:
                L_style = NST.sum_style_losses(sess, net, style_imgs, opt)

            # content loss
            L_content = NST.sum_content_losses(sess, net, content_img, opt)

            # denoising loss
            L_tv = tf.image.total_variation(net['input'])

            # loss weights
            alpha = opt.content_weight
            beta = opt.style_weight
            theta = opt.tv_weight

            # total loss
            L_total = alpha * L_content
            L_total += beta * L_style
            L_total += theta * L_tv

            # optimization algorithm
            optimizer = NST.get_optimizer(L_total, opt)

            if opt.optimizer == 'adam':
                NST.minimize_with_adam(sess, net, optimizer, init_img, L_total, opt)
            elif opt.optimizer == 'lbfgs':
                NST.minimize_with_lbfgs(sess, net, optimizer, init_img, opt)

            output_img = sess.run(net['input'])

            if opt.original_colors:
                output_img = NST.convert_to_original_colors(np.copy(content_img), output_img, opt)

            sess.close()

            return output_img

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def minimize_with_lbfgs(sess, net, optimizer, init_img, opt):
        if opt.verbose:
            print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)
        sess.run(net['input'].assign(init_img))
        optimizer.minimize(sess)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def minimize_with_adam(sess, net, optimizer, init_img, loss, opt):
        if opt.verbose:
            print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
        train_op = optimizer.minimize(loss)
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)
        sess.run(net['input'].assign(init_img))
        iterations = 0
        while iterations < opt.max_iterations:
            sess.run(train_op)
            if iterations % opt.print_iterations == 0 and opt.verbose:
                curr_loss = loss.eval()
                print("At iterate {}\tf=  {}".format(iterations, curr_loss))
            iterations += 1

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_optimizer(loss, opt):
        print_iterations = opt.print_iterations if opt.verbose else 0
        if opt.optimizer == 'lbfgs':
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss, method='L-BFGS-B',
                options={'maxiter': opt.max_iterations,
                         'disp': print_iterations})
        elif opt.optimizer == 'adam':
            optimizer = tf.compat.v1.train.AdamOptimizer(opt.learning_rate)
        return optimizer

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_mask_image(mask_img, width, height, opt):
        path = os.path.join(opt.content_img_dir, mask_img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        mx = np.amax(img)
        img /= mx
        return img

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def convert_to_original_colors(content_img, stylized_img, opt):
        content_img = NST.postprocess(content_img)
        stylized_img = NST.postprocess(stylized_img)
        if opt.color_convert_type == 'yuv':
            cvt_type = cv2.COLOR_BGR2YUV
            inv_cvt_type = cv2.COLOR_YUV2BGR
        elif opt.color_convert_type == 'ycrcb':
            cvt_type = cv2.COLOR_BGR2YCR_CB
            inv_cvt_type = cv2.COLOR_YCR_CB2BGR
        elif opt.color_convert_type == 'luv':
            cvt_type = cv2.COLOR_BGR2LUV
            inv_cvt_type = cv2.COLOR_LUV2BGR
        elif opt.color_convert_type == 'lab':
            cvt_type = cv2.COLOR_BGR2LAB
            inv_cvt_type = cv2.COLOR_LAB2BGR
        content_cvt = cv2.cvtColor(content_img, cvt_type)
        stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
        c1, _, _ = cv2.split(stylized_cvt)
        _, c2, c3 = cv2.split(content_cvt)
        merged = cv2.merge((c1, c2, c3))
        dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
        dst = NST.preprocess(dst)
        return dst

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def render_single_image(content_img, style_img, opt):
        with tf.Graph().as_default():
            print('\n---- RENDERING SINGLE IMAGE ----\n')
            tick = time.time()
            output_img = NST.stylize(content_img, style_img, content_img, opt)
            tock = time.time()
            print('Single image elapsed time: {}'.format(tock - tick))
            return output_img
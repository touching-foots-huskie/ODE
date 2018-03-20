# -*- coding:utf-8 -*-
#  one order differential eqaution.
import numpy as np
import tensorflow as tf


class ODE_solver:
    def __init__(self, f, x0, y0):
        self.f = f
        self.x0 = x0
        self.y0 = y0
        self.network()
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def network(self):
        self.x = tf.placeholder(tf.float32, [None, 1])
        #  3 layers:
        self.n = tf.layers.dense(self.x, 6, tf.nn.leaky_relu)
        self.n = tf.layers.dense(self.n, 6, tf.nn.leaky_relu)
        self.n = tf.layers.dense(self.n, 6, tf.nn.leaky_relu)
        self.yp = self.y0 + (self.x - self.x0)*self.n

        self.gradient = tf.gradients(self.yp, self.x)
        

if __name__ == '__main__':
    def f(x):
        return x^2

    solver = ODE_solver(f, 0, 1)



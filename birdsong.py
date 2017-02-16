#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for training generative model for birdsong.
"""

import argparse
import os

import model
import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Code for training birdsong generator model.')

    parser.add_argument('-d', '--rebuild-data',
            default=False,
            action='store_true',
            help='If set, rebuilds the dataset.')
    parser.add_argument('-m', '--rebuild-model',
            default=False,
            action='store_true',
            help='If set, resets the model weigts.')
    parser.add_argument('-r', '--plot-real',
            default=False,
            action='store_true',
            help='If set, plot samples of real data.')
    parser.add_argument('-g', '--plot-gen',
            default=False,
            action='store_true',
            help='If set, plot samples of generated data.')
    parser.add_argument('-G', '--plot-gif',
            default=False,
            action='store_true',
            help='If set, make a gif of interpolating latent space.')
    parser.add_argument('-t', '--time-length',
            default=50,
            type=int,
            metavar='N',
            help='Number of bins in the time axis per sample.')
    parser.add_argument('-n', '--nb-epoch',
            default=10,
            type=int,
            metavar='N',
            help='Number of epochs to train.')

    args = parser.parse_args()

    x = utils.get_all_spectrograms(args.time_length, rebuild=args.rebuild_data)

    if args.plot_real:
        utils.plot_sample(x)

    trained_model = model.train(x,
            nb_epoch=args.nb_epoch,
            rebuild=args.rebuild_model)

    if args.plot_gen:
        x = trained_model.sample(['normal'], num_samples=32)
        utils.plot_sample(x)

    if args.plot_gif:
        pts = model.interpolate_latent_space(trained_model, nb_points=60)
        utils.plot_as_gif(pts)

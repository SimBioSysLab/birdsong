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

    parser.add_argument('--rebuild', default=False, action='store_true',
                        help='If set, rebuilds the dataset.')
    parser.add_argument('--time_length', default=300, type=int, metavar='N',
                        help='Number of bins in the time axis per sample.')

    args = parser.parse_args()

    x = utils.get_all_spectrograms(args.time_length, rebuild=args.rebuild)

    # For plotting.
    # utils.plot_sample(x)

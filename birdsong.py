#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some generative models for birdsong.
"""

import argparse
import os

import utils


if __name__ == '__main__':
    x = utils.get_all_spectrograms(100)

    utils.plot_sample(x, shuffle=False, downsample=3)

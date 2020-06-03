#!/usr/bin/env python3
import click

from ..embeddings import build_embed_matrix, load_glove
from ..lstm import lstm


@click.command()
@click.option('-v', '--vocab', required=True, help='vocabulary size')
@click.option('-d', '--dim', required=False, help='embedding dimension')
@click.option('-h', '--hidden', required=True, help='number of hidden units')
@click.option('-w', '--weights', required=False, type=click.Choice(['glove']), help='pretrained embedding weights')
@click.option('-te', required=False, type=bool, help='whether to train embedding layer')
def train(vocab_size, embed_dim, units, weights=None, train_embed=False):
    # load data
    # preprocess data
    # build tokenizer
    # form sequnces
    # pad sequences
    # split
    # optional: load embedding weights
    # build model
    # compile model
    # train model
    # save model

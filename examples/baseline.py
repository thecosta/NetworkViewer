import sys
sys.path.append("..")

from core import Layer, Tuple
from plotter import Plotter


layers = [
    Layer(name='input', output_dim=Tuple(25, 198), depth=2,
          output_dim_label='(S, 2)'),
    Layer(name='conv1d_1', output_dim=Tuple(25, 198), depth=128, activation='ReLu',
          output_dim_label='(S, 128)'),
    Layer(name='conv1d_2', output_dim=Tuple(25, 198), depth=128, activation='ReLu', maxpool=True,
          output_dim_label='(S, 128)'),
    Layer(name='conv1d_3', output_dim=Tuple(25, 99), depth=128, activation='ReLu',
          output_dim_label='(S, 128)'),
    Layer(name='conv1d_4', output_dim=Tuple(25, 99), depth=128, activation='ReLu', maxpool=True,
          output_dim_label='(S, 128)'),
    Layer(name='conv1d_5', output_dim=Tuple(25, 49), depth=128, activation='ReLu',
          output_dim_label='(S, 128)'),
    Layer(name='conv1d_6', output_dim=Tuple(25, 49), depth=128, activation='ReLu', maxpool=True,
          output_dim_label='(S, 128)'),
    Layer(name='conv1d_7', output_dim=Tuple(25, 24), depth=128, activation='ReLu',
          output_dim_label='(S, 128)'),
    Layer(name='conv1d_8', output_dim=Tuple(25, 24), depth=128, activation='ReLu', maxpool=True,
          output_dim_label='(S, 128)'),
    Layer(name='conv1d_9', output_dim=Tuple(25, 12), depth=128, activation='ReLu',
          output_dim_label='(S, 128)'),
    Layer(name='conv1d_10', output_dim=Tuple(25, 12), depth=128, activation='ReLu', flatten=True,
          output_dim_label='(S, 128)'),
    Layer(name='dense_1', output_dim=Tuple(25, 1), depth=256, activation='ReLu',
          output_dim_label='(256)'),
    Layer(name='dense_2', output_dim=Tuple(25, 1), depth=128, activation='ReLu',
          output_dim_label='(128)'),
    Layer(name='dense_3', output_dim=Tuple(25, 1), depth=50, activation='Softmax',
          output_dim_label='(50)')]

Plotter(layers=layers, window_height=500, window_width=2000, padding=150, title='Baseline Model')

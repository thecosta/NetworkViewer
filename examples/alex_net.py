import sys
sys.path.append("..")

from core import Layer, Tuple
from plotter import Plotter


layers = [
    Layer(name='conv1d_1', output_dim=Tuple(55, 55), depth=90,
          maxpool=True, activation='ReLu', output_dim_label='(55, 55, 96)'),
    Layer(name='conv1d_2', output_dim=Tuple(27, 27), depth=200,
          maxpool=True, activation='ReLu', output_dim_label='(27, 27, 256)'),
    Layer(name='conv1d_3', output_dim=Tuple(13, 13), depth=300, activation='ReLu',
          output_dim_label='(13, 13, 384)'),
    Layer(name='conv1d_4', output_dim=Tuple(13, 13), depth=300, activation='ReLu',
          output_dim_label='(13, 13, 384)'),
    Layer(name='conv1d_5', output_dim=Tuple(13, 13), depth=200, 
          maxpool=True, flatten=True, activation='ReLu', output_dim_label='(13, 13, 256)'),
    Layer(name='dense_1', output_dim=Tuple(25, 1), depth=921, activation='ReLu',
          output_dim_label='(9216)'),
    Layer(name='dense_2', output_dim=Tuple(25, 1), depth=409, activation='ReLu',
          output_dim_label='(4096)'),
    Layer(name='dense_3', output_dim=Tuple(25, 1), depth=409, activation='ReLu',
          output_dim_label='(4096)'),
    Layer(name='dense_4', output_dim=Tuple(25, 1), depth=100, activation='Softmax',
          output_dim_label='(1000)')]

Plotter(layers=layers, window_height=400, window_width=800, padding=50, title='Alex Net', depth_scale=.25)

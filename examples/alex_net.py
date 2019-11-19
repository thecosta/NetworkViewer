import sys
sys.path.append("..")

from core import Layer, Tuple
from plotter import Plotter


layers = [
    Layer(name='input', output_dim=Tuple(227, 227), depth=3, output_dim_label='(227, 227, 3)'),
    Layer(name='conv1d_1', output_dim=Tuple(55, 55), depth=96,
          maxpool=True, activation='relu', output_dim_label='(55, 55, 96)'),
    Layer(name='conv1d_2', output_dim=Tuple(27, 27), depth=256,
          maxpool=True, activation='relu', output_dim_label='(27, 27, 256)'),
    Layer(name='conv1d_3', output_dim=Tuple(13, 13), depth=384, activation='relu',
          output_dim_label='(13, 13, 384)'),
    Layer(name='conv1d_4', output_dim=Tuple(13, 13), depth=384, activation='relu',
          output_dim_label='(13, 13, 384)'),
    Layer(name='conv1d_5', output_dim=Tuple(13, 13), depth=256, 
          maxpool=True, flatten=True, activation='relu', output_dim_label='(13, 13, 256)'),
    Layer(name='dense_1', output_dim=Tuple(25, 1), depth=9216, activation='relu',
          output_dim_label='(9216)'),
    Layer(name='dense_2', output_dim=Tuple(25, 1), depth=4096, activation='relu',
          output_dim_label='(4096)'),
    Layer(name='dense_3', output_dim=Tuple(25, 1), depth=4096, activation='relu',
          output_dim_label='(4096)'),
    Layer(name='dense_4', output_dim=Tuple(25, 1), depth=1000, activation='softmax',
          output_dim_label='(1000)'),
    Layer(name='output', output_dim=Tuple(25, 1), depth=1,
          output_dim_label='(1)')]

Plotter(layers=layers, window_height=500, window_width=2000, padding=150, title='Alex Net', depth_scale=0.05)

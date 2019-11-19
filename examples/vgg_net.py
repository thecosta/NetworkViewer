import sys
sys.path.append("..")
from core import Layer, Tuple
from plotter import Plotter


layers = [
    
    Layer(name='conv1d_1', output_dim=Tuple(x=25, y=224), depth=64, activation='relu',
          output_dim_label='(224, 64)'),
    Layer(name='conv1d_2', output_dim=Tuple(x=25, y=224), depth=64, activation='relu', maxpool=True,
          output_dim_label='(224, 64)'),
    
    Layer(name='conv1d_3', output_dim=Tuple(x=25, y=112), depth=128, activation='relu',
          output_dim_label='(112, 128)'),
    Layer(name='conv1d_4', output_dim=Tuple(x=25, y=112), depth=128, activation='relu', maxpool=True,
          output_dim_label='(112, 128)'),
    
    Layer(name='conv1d_5', output_dim=Tuple(x=25, y=56), depth=256, activation='relu',
          output_dim_label='(56, 256)'),
    Layer(name='conv1d_6', output_dim=Tuple(x=25, y=56), depth=256, activation='relu', maxpool=True,
          output_dim_label='(56, 256)'),
    
    Layer(name='conv1d_7', output_dim=Tuple(x=25, y=28), depth=512, activation='relu',
          output_dim_label='(28, 512)'),
    Layer(name='conv1d_8', output_dim=Tuple(x=25, y=28), depth=512, activation='relu',
          output_dim_label='(28, 512)'),
    Layer(name='conv1d_9', output_dim=Tuple(x=25, y=28), depth=512, activation='relu', maxpool=True,
          output_dim_label='(28, 512)'),
    
    Layer(name='conv1d_10', output_dim=Tuple(x=25, y=14), depth=512, activation='relu',
          output_dim_label='(14, 512)'),
    Layer(name='conv1d_11', output_dim=Tuple(x=25, y=14), depth=512, activation='relu',
          output_dim_label='(14, 512)'),
    Layer(name='conv1d_12', output_dim=Tuple(x=25, y=14), depth=512, activation='relu', maxpool=True, flatten=True,
          output_dim_label='(14, 512)'),
    
    Layer(name='dense_1', output_dim=Tuple(x=25, y=7), depth=4096, activation='relu',
          output_dim_label='(7, 4096)'),
    Layer(name='dense_2', output_dim=Tuple(x=25, y=7), depth=4096, activation='relu',
          output_dim_label='(7, 4096)'),
    Layer(name='dense_3', output_dim=Tuple(x=25, y=7), depth=4096, activation='softmax',
          output_dim_label='(7, 4096)'),

    Layer(name='output', output_dim=Tuple(x=25, y=1), depth=1,
          output_dim_label='(1)')]

Plotter(layers=layers, window_height=700, window_width=2500, padding=100, title='VGG Net', depth_scale=0.05)

import h5py
import matplotlib
import matplotlib.collections as collections
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np

from core import Tuple, Layer
from layers import Convolution, Dense, Input, Funnel, Output
from matplotlib.pyplot import figure
from matplotlib.collections import PatchCollection


class Plotter():
    '''Plot neural networks.'''

    def __init__(self, layers, window_width=1000, window_height=700,
                 dpi=100, padding=None, show_dim=True, show_lower_text=True,
                 title=None, x_scale=1, y_scale=1, depth_scale=1, log=False):
        '''Initializer.
        
        Attributes:
        -----------
        
            layers: list of Layer.
            window_width: pixel width of plotting window.
            window_height: pixel height of plotting window.
            dpi: dots per inch.
            padding: pixel padding between layers.
            show_dim: show layer dimensions.
            show_lower_text: show lower layer text, ie activation functions
                             maxpooling, flatten.
            title: plot title.
            x_scale: scale all x dimensions.
            y_scale: scale all y dimensions.
            depth_scale: scale all depth dimensions.
            log: scale by log factor all dimensions.
        '''
        self.layers = layers
        self.window_height = window_height
        self.window_width = window_width
        self.dpi = dpi
        self.padding = padding
        self.show_dim = show_dim
        self.show_lower_text = show_lower_text
        self.title = title
        if x_scale < 1:
            for layer in layers:
                layer.output_dim.x = int(layer.output_dim.x*x_scale)
        if y_scale < 1:
            for layer in layers:
                layer.output_dim.y = int(layer.output_dim.y*y_scale)
        if log:
            for layer in layers:
                layer.depth = int(np.log(layer.depth))
                layer.output_dim.x = int(np.log(layer.output_dim.x))
                layer.output_dim.y = int(np.log(layer.output_dim.y))
        else:
            if depth_scale < 1:
                for layer in layers:
                    layer.depth = int(layer.depth*depth_scale)+1     
        self.layout()
        self.get_shapes()
        self.view_window()
      
    
    def layout(self):
        '''Set layers coordinates in window.'''
        total_layer_width = 0.0
        for layer in self.layers:
                total_layer_width += layer.output_dim.x
                
        num_layers = len(self.layers)
        total_padding = self.window_width - total_layer_width
        
        if self.window_width < total_layer_width:
            raise ValueError('Use larger window width! ' + str(self.window_width) + ' < ' + str(total_layer_width))
        if self.window_width < (total_layer_width + self.padding*(num_layers+1)):
            print('Warning: updating padding: ', 
                  (self.window_width - total_layer_width) / (num_layers+1), 
                  ', ', self.window_width, 
                  ' < ', total_layer_width + self.padding*(num_layers+1))
            self.padding = (self.window_width - total_layer_width) / (num_layers+1)
            
        current_x = self.padding
        self.largest_y = 0
        
        # Get X-coordinates for layers
        for i, layer in enumerate(self.layers):
            layer.position = Tuple(current_x, 0)
            #adjust = self.padding*(self.layers[i-1].depth-layer.depth)/(4*self.padding)
            #if i and adjust > 0:
            #    layer.position = Tuple(current_x+adjust, 0)
            #    print('i: ', i, ', ', adjust)
            #else:
            current_x += layer.output_dim.x + self.padding
            height = layer.output_dim.y + layer.depth
            if height > self.largest_y:
                self.largest_y = height
        
        # Get Y-coordinates for layers
        for layer in self.layers:
            layer.position.y = 0
            
        if self.window_height < self.largest_y+100+self.padding:
            self.window_height = self.largest_y+100+self.padding
            print('Adjusting window height to recommended size: ', self.window_height)
            
        if self.window_width < total_layer_width + self.padding*(num_layers+2):
            self.window_width = total_layer_width + self.padding*(num_layers+2)
            print('Adjusting window width to recommended size: ', self.window_width)
        
    def get_shapes(self):
        '''Draw layer shapes.'''
        shapes = []
        for i, layer in enumerate(self.layers):
            if 'input' in layer.name:
                shapes = np.append(shapes, 
                                   Input(name=layer.name,
                                         position=layer.position,
                                         output_dim=layer.output_dim,
                                         depth=layer.depth,
                                         flatten=layer.flatten,
                                         output_dim_label=layer.output_dim_label).draw())
            if 'dense' in layer.name:
                shapes = np.append(shapes, 
                                   Dense(name=layer.name,
                                         position=layer.position,
                                         output_dim=layer.output_dim,
                                         depth=layer.depth,
                                         activation=layer.activation,
                                         maxpool=layer.maxpool,
                                         flatten=layer.flatten,
                                         output_dim_label=layer.output_dim_label).draw())
            if 'conv' in layer.name:
                shapes = np.append(shapes, 
                                   Convolution(name=layer.name,
                                               position=layer.position,
                                               output_dim=layer.output_dim,
                                               depth=layer.depth,
                                               activation=layer.activation,
                                               maxpool=layer.maxpool,
                                               flatten=layer.flatten,
                                               output_dim_label=layer.output_dim_label).draw())
            if 'output' in layer.name:
                shapes = np.append(shapes, 
                                   Output(name=layer.name,
                                         position=layer.position,
                                         output_dim=layer.output_dim,
                                         depth=layer.depth,
                                         output_dim_label=layer.output_dim_label).draw())
            if i:
                shapes = np.append(shapes,
                                   Funnel(prev_position=self.layers[i-1].position,
                                          prev_depth=self.layers[i-1].depth,
                                          prev_output_dim=self.layers[i-1].output_dim,
                                          curr_position=layer.position,
                                          curr_depth=layer.depth,
                                          curr_output_dim=layer.output_dim,
                                          color=(178.0/255, 178.0/255, 178.0/255)).draw())
        self.shapes = shapes
        
            
    def view_window(self):
        '''Initialize figure window.'''
        fig, ax = plt.subplots(figsize=(self.window_width/self.dpi, self.window_height/self.dpi),
                               dpi=self.dpi)
        [ax.add_patch(shape) for shape in self.shapes]      
        ax.set_xlim((0, self.window_width))
        ax.set_ylim((-150, self.window_height))
        ax.figure.canvas.draw()
        ax.autoscale_view()
        
        for layer in self.layers:
            if self.show_dim:
                if layer.output_dim_label:
                    upper_text = layer.output_dim_label
                else:
                    upper_text = '('+str(layer.output_dim.y)+', '+str(layer.depth)+')'
                    
                x = layer.position.x + layer.output_dim.x/2 + layer.depth
                y = layer.position.y + layer.output_dim.y + 10 + layer.depth
                ax.annotate(upper_text, xy=(200, 100), xytext=(x, y),
                            rotation=45, fontsize=7)
            if self.show_lower_text:
                lower_text = ''
                #lower_text += layer.output_dim_label + '\n'
                if layer.activation:
                    lower_text += layer.activation + '\n'
                if layer.maxpool:
                    lower_text += 'Maxpool\n'
                if layer.flatten:
                    lower_text += 'Flatten\n'
                start_x = layer.position.x+layer.output_dim.x/2
                start_y = layer.position.y
                end_x = layer.position.x+layer.output_dim.x/2 - 10
                end_y = layer.position.y - 20
                ax.annotate(lower_text,
                            xy=(start_x,start_y),
                            xytext=(end_x, end_y),
                            rotation_mode='anchor',
                            verticalalignment='top',
                            horizontalalignment='left',
                            rotation=0,
                            fontsize=7)
                
                
        plt.title(self.title)        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        input_patch = pat.Patch(color=(207.0/255, 242.0/255, 212.0/255), label='Input')
        conv_patch = pat.Patch(color=(255.0/255, 242.0/255, 204.0/255), label='1D Convolution Layer')
        fc_patch = pat.Patch(color=(242.0/255, 207.0/255, 207.0/255), label='Fully Connected Layer')
        ax.legend(handles=[input_patch, conv_patch, fc_patch], ncol=3, loc='upper center')
        plt.show()
        
    
if __name__ == '__main__':
    Plotter('./weights/weights_baseline.hdf5')

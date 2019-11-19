import numpy as np

class Tuple():
    '''Represent a tuple.'''
    
    def __init__(self, x, y):
        '''Initializer.'''
        self.x = x
        self.y = y
        
    def get_tuple(self):
        '''Class representation.'''
        return (self.x, self.y)
    
    def __str__(self):
        '''String representation.'''
        return str((self.x, self.y))
    
    def __mul__(self, value):
        '''Object multiplication.'''
        return Tuple(int(self.x*value), int(self.y*value))
    
        
class Layer():
    '''A neural network layer.'''
    
    def __init__(self, name, output_dim=Tuple(None, None), depth=None,
                 position=Tuple(None, None), activation=None,
                 maxpool=False, flatten=False, output_dim_label=None):
        '''Initializer.
        
        Attributes:
        -----------
            name: layer name. must include 'dense', 'conv', 'input', or 'output'
                  if layer represents one of such.
            output_dim: dimensionality of the output layer.
            depth: depth or number of channels of the output layer.
            output_dim_label: output dimensionality label to use. useful for when
                              reshaping output dimensions. 
        '''
        self.name = name
        self.output_dim = output_dim
        self.depth = depth
        self.position = position
        self.activation = activation
        self.maxpool = maxpool
        self.flatten = flatten
        self.output_dim_label = output_dim_label
        
    def __str__(self):
        '''String representation.'''
        return 'Layer'+str(tuple((self.name, str(self.output_dim), self.depth, str(self.position))))
    
    
class Kernel(Layer):
    '''A layer's weights.'''
    
    def __init__(self, name, output_dim, position, weights):
        '''Initializer.'''
        super().__init__(name=name, output_dim=output_dim, position=position, weights=weights)
    
    def __str__(self):
        '''String representation.'''
        return 'Layer'+str(tuple((self.name, str(self.output_dim), self.depth, str(self.position))))
    
        
class Color:
    '''Color vector for weights.'''
    
    def __init__(self, weights):
        '''Initializer.'''
        self.color = [0, 0, 255]
        self.weights = weights

    
    def colorize(self):
        if self.weights is None:
            return None
        colors = []
        min_value = abs(np.min(self.weights))
        max_value = np.max(self.weights) + abs(min_value)
        [colors.append(((1.0*weight+min_value)/max_value,
                        (1.0*weight+min_value)/max_value, 1)) for i, weight in enumerate(self.weights)]
        return colors 
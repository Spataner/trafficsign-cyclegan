'''
CycleGAN architecture classes.

Notice: These classes support three extensions to the CycleGAN system that were ultimately not used in the thesis.
"info_gan": True                    -- InfoGAN extension (https://arxiv.org/pdf/1606.03657.pdf)
"class_mode": "class_conditional"   -- Class label information is passed into mappings and discriminators alongside an image.
"class_mode": "class_loss"          -- Discriminator must output correct data class, using either an n+1 scheme (n real
                                       classes and one fake class) or a 2n scheme (n real classes and n fake classes).

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import enum
import tensorflow                   as tf
import cyclegan.model.operations    as ops
import cyclegan.util.config         as cfg
import cyclegan.util.exceptions     as exc


class Graph:
    '''
    Base class for CycleGAN graph elements.
    '''

    def __init__(self, name):
        self.name = name

    def create(self, input, reuse):
        '''
        Creates the variable scope and graph structure for one instance of this graph element.

        Arguments:
            input       -- The input tensor for this instance of the graph.
            reuse       -- Whether this instance shall reuse variables from a previous one.

        Returns:
            The graph instance's output tensor.
        '''

        with tf.variable_scope(self.name, reuse = reuse):
            return self.build_graph(input)

    def build_graph(self, input):
        '''
        Builds the graph structure for one instance of this graph element.
        For internal use only.

        Arguments:
            input   -- The input tensor for the graph.

        Returns:
            The output tensor of the graph.
        '''

        return input

    def get_variables(self):
        '''
        Retrieve all trainable variables from this graph element's scope.

        Returns:
            A list of trainable variables.
        '''
        
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)

    def parameter_count(self):
        return sum((variable.shape.num_elements() for variable in self.get_variables()))


class Discriminator(Graph):
    '''
    Discriminator base class for CycleGAN architecture.
    '''

    SLOPE = 0.2
    KERNEL_SIZE = [4, 4]

    def __init__(self, name, instance_norm_affine, initial_filter_count, target_channels = 1):
        '''
        Setter constructor.

        Arguments:
            name                    -- The variable scope name for the discriminator's graph.
            initial_filter_count    -- The number of filters in the first layer of the discriminator. Later layers use
                                       multiples of this value.
            instance_norm_affine    -- Whether the instance norm layers shall contain a trainable affine transform.
            target_channels         -- The number of image channels in the final layer of the discriminator.
        '''

        Graph.__init__(self, name)
        self.instance_norm_affine = instance_norm_affine
        self.target_channels = target_channels
        self.filter_counts = [
            initial_filter_count, initial_filter_count * 2,
            initial_filter_count * 4, initial_filter_count * 8
        ]

    def build_graph(self, input):
        '''
        Creates one instance of the discriminator's Tensorflow graph.

        Arguments:
            input   -- The input tensor for this instance of the graph.

        Returns:
            The output tensor of the graph.
        '''

        conv_down1 = ops.conv2d_downsample_leaky(input, self.KERNEL_SIZE, self.filter_counts[0], self.SLOPE, instance_norm = False)
        conv_down2 = ops.conv2d_downsample_leaky(conv_down1, self.KERNEL_SIZE, self.filter_counts[1], self.SLOPE, instance_norm_affine = self.instance_norm_affine)
        conv_down3 = ops.conv2d_downsample_leaky(conv_down2, self.KERNEL_SIZE, self.filter_counts[2], self.SLOPE, instance_norm_affine = self.instance_norm_affine)
        
        conv1 = ops.conv2d(
            conv_down3, self.KERNEL_SIZE, self.filter_counts[3], activation = lambda x: tf.nn.leaky_relu(x, alpha = self.SLOPE),
            instance_norm_affine = self.instance_norm_affine
        )
        conv2 = ops.conv2d(conv1, self.KERNEL_SIZE, self.target_channels, instance_norm = False, activation = None)

        self.layers = [conv_down1, conv_down2, conv_down3, conv1, conv2]

        return conv2


class Mapping(Graph):
    '''
    Mapping base class for CycleGAN architecture.
    '''

    CONV_KERNEL_SIZE = [7, 7]
    DOWN_KERNEL_SIZE = [3, 3]
    RES_KERNEL_SIZE = [3, 3]
    UP_KERNEL_SIZE = [3, 3]

    def __init__(self, name, res_block_count, res_block_relu, channel_count, intial_filter_count, instance_norm_affine):
        '''
        Setter constructor.

        Arguments:
            name                    -- The name of the variable scope of the mapping's graph.
            res_block_count         -- The number of residual blocks to use.
            res_block_relu          -- Whether residual blocks shall use ReLU activation on their output.
            channel_count           -- The channel count of the target domain.
            initial_filter_count    -- The number of filters in the first layer of the mapping. Later layers use
                                       multiples of this value.
            instance_norm_affine    -- Whether the instance norm layers shall contain a trainable affine transform.
        '''

        Graph.__init__(self, name)

        self.res_block_count = res_block_count
        self.res_block_relu = res_block_relu
        self.channel_count = channel_count
        self.instance_norm_affine = instance_norm_affine
        self.filter_counts = [intial_filter_count, intial_filter_count * 2, intial_filter_count * 4]

    def build_graph(self, input):
        '''
        Creates one instance of the mapping's Tensorflow graph.

        Arguments:
            input   -- The input tensor for this instance of the graph.

        Returns:
            The output tensor of the graph.
        '''

        #Downsample
        conv1 = ops.conv2d(input, self.CONV_KERNEL_SIZE, self.filter_counts[0], instance_norm_affine = self.instance_norm_affine)
        conv_down1 = ops.conv2d_downsample(conv1, self.DOWN_KERNEL_SIZE, self.filter_counts[1], instance_norm_affine = self.instance_norm_affine)
        conv_down2 = ops.conv2d_downsample(conv_down1, self.DOWN_KERNEL_SIZE, self.filter_counts[2], instance_norm_affine = self.instance_norm_affine)


        #Residual blocks

        res_blocks = list()
        next_input = conv_down2

        res_block_layers = [
            lambda x : ops.conv2d(x, self.RES_KERNEL_SIZE, self.filter_counts[2], instance_norm_affine = self.instance_norm_affine),
            lambda x : ops.conv2d(x, self.RES_KERNEL_SIZE, self.filter_counts[2], activation = None, instance_norm_affine = self.instance_norm_affine)
        ]

        for _ in range(self.res_block_count):
            res_blocks.append(ops.residual_block(next_input, res_block_layers, activation = tf.nn.relu if self.res_block_relu else None))
            next_input = res_blocks[-1]


        #Upsample
        conv_up1 = ops.conv2d_upsample(res_blocks[-1], self.UP_KERNEL_SIZE, self.filter_counts[1], instance_norm_affine = self.instance_norm_affine)
        conv_up2 = ops.conv2d_upsample(conv_up1, self.UP_KERNEL_SIZE, self.filter_counts[0], instance_norm_affine = self.instance_norm_affine)
        conv2 = ops.conv2d(conv_up2, self.CONV_KERNEL_SIZE, self.channel_count, instance_norm = False, activation = tf.tanh)

        self.layers = [conv1, conv_down1, conv_down2]
        self.layers += res_blocks
        self.layers += [conv_up1, conv_up2, conv2]

        return conv2


class VectorDecorator(Graph):
    '''
    Decorator for a graph element class that conditions its input on additional input vectors.
    It does so by assuming that its input is a list of one image and an arbitrary number of vectors.
    It will optionally project the vectors into an image-shape and concatenate them to the input image along the channel dimension.
    It may furthermore pass through these vectors to the output of the graph element.
    This stacked representation of the input is then provided to the actual graph element to build the graph.
    '''

    PROJECTION_CHANNELS = 3

    def __init__(self, Class, concat, passthrough, *args, **kws):
        '''
        Setter constructor.

        Arguments:
            Class       -- The class that this object decorates.
            concat      -- A list of boolean values of same length as the list of additional vectors,
                           indicating whether each vector should be projected/reshaped/concatenated to the image.
            passthrough -- A list of boolean values of same length as the list of additional vectors,
                           indicating whether each vector should be passed through to the output.
        '''

        self.element = Class(*args, **kws)

        self.concat = concat
        self.passthrough = passthrough

        Graph.__init__(self, self.element.name)

    def build_graph(self, input):
        image = input[0]
        vectors = input[1:]

        reshaped = list()

        for i in range(len(vectors)):
            if not self.concat[i]: continue

            projected = tf.layers.dense(vectors[i], self.PROJECTION_CHANNELS * int(image.shape[1]) * int(image.shape[2]))
            reshaped.append(tf.reshape(projected, [-1, int(image.shape[1]), int(image.shape[2]), self.PROJECTION_CHANNELS]))

        image = tf.concat([image] + reshaped, 3)

        outputs = [vectors[i] for i in range(len(vectors)) if self.passthrough[i]]
        output = tuple([self.element.build_graph(image)] + outputs)

        self.layers = reshaped + self.element.layers

        return output


class InfoGANDecorator(Graph):
    '''
    Decorates a discriminator class which a secondary infoGAN head attached to the penultimate discriminator layer
    that is meant to reconstruct an additional latent variable vector.
    '''

    def __init__(self, Class, output_size, *args, **kws):
        '''
        Setter constructor.

        Arguments:
            Class       -- The discriminator class to decorate.
            output_size -- The size of the secondary heads output tensor (same as the size of the latent variable vector).
        '''

        self.element = Class(*args, **kws)

        self.output_size = output_size

        Graph.__init__(self, self.element.name)

    def build_graph(self, input):
        output = self.element.build_graph(input)

        flattened = tf.layers.flatten(self.element.layers[-2])
        secondary_head = tf.layers.dense(flattened, self.output_size)

        self.layers = self.element.layers + [secondary_head]

        return output + (secondary_head,)


class Mode(enum.Enum):
    '''
    Enum class for architecture modes.
    '''

    TRAIN   = 0   #Build full graph of mappings and discriminators.
    TEST    = 1   #Build only mappings but both input and cycle versions.
    RUN     = 2   #Build only input versions of the mappings.
    RUN_XY  = 3   #Build only input version of the X -> Y mapping.
    RUN_YX  = 4   #Build only input version of the Y -> X mapping.


class TupleWrapper:
    '''
    Wrapper for ease of access to graph element outputs that may or may not be a tuple.
    "self.main" always contains the main output (image or matrix), "self.passthroughs" any additionaly vectors
    that were passed along.
    '''

    def __init__(self, output_node):
        if type(output_node) == tuple:
            self.all = output_node
            self.main = output_node[0]
            self.passthroughs = output_node[1:]
        else:
            self.all = output_node
            self.main = output_node
            self.passthroughs = tuple()

    def __getitem__(self, index):
        if type(self.all) != tuple:
            raise exc.ArchitectureException.incorrect_indexed_access()

        return self.all[index]


class CycleGAN:
    '''
    CycleGAN architecture class.
    Creates and holds the Tensorflow graphs for the CycleGAN.
    As according to Zhu et al. (https://arxiv.org/pdf/1703.10593.pdf).
    '''

    #String constants
    MAPPING_XY_NAME = 'mapping_xy'
    MAPPING_YX_NAME = 'mapping_yx'
    DISCRIMINATOR_X_NAME = 'discriminator_x'
    DISCRIMINATOR_Y_NAME = 'discriminator_y'

    def __init__(self, MappingClass, DiscriminatorClass, config, mode = Mode.TRAIN):
        '''
        Setter constructor.

        Arguments:
            MappingClass        -- The class for creating the mapping objects.
            DiscriminatorClass  -- The class for creating the discriminator objects.
            config              -- The CycleGAN's configuration.
            mode                -- The architecture mode.
        '''

        self.MappingClass = MappingClass
        self.DiscriminatorClass = DiscriminatorClass
        self.config = config
        self.mode = mode

    def create_mappings(self, input_x, input_y):
        '''
        Creates the two mappings (X -> Y and Y -> X).

        Arguments:
            input_x -- The input of real images for domain X.
            input_y -- The input of real images for domain Y.

        Returns:
            The two mapping objects.
        '''

        #Create the mapping graph versions with the respective input data as input.

        if self.mode != Mode.RUN_YX:
            mapping_xy = self.MappingClass(
                self.MAPPING_XY_NAME, self.config[cfg.RES_BLOCK_COUNT], self.config[cfg.RES_BLOCK_RELU],
                self.config[cfg.CHANNEL_COUNT_Y], self.config[cfg.MAPPING_XY_INITIAL_FILTER_COUNT],
                self.config[cfg.INSTANCE_NORM_AFFINE]
            )

            mapping_xy.from_input = TupleWrapper(mapping_xy.create(input_x, False))
        else:
            mapping_xy = None

        if self.mode != Mode.RUN_XY:
            mapping_yx = self.MappingClass(
                self.MAPPING_YX_NAME, self.config[cfg.RES_BLOCK_COUNT], self.config[cfg.RES_BLOCK_RELU],
                self.config[cfg.CHANNEL_COUNT_X], self.config[cfg.MAPPING_XY_INITIAL_FILTER_COUNT],
                self.config[cfg.INSTANCE_NORM_AFFINE]
            )

            mapping_yx.from_input = TupleWrapper(mapping_yx.create(input_y, False))
        else:
            mapping_yx = None

        if self.mode in (Mode.TRAIN, Mode.TEST):
            #Create the mapping graph versions with each other as input.
            mapping_xy.cycle = TupleWrapper(mapping_xy.create(mapping_yx.from_input.all, True))
            mapping_yx.cycle = TupleWrapper(mapping_yx.create(mapping_xy.from_input.all, True))

        return mapping_xy, mapping_yx

    def create_discriminators(self, input_x, input_y):
        '''
        Creates one discriminator for each domain.

        Arguments:
            input_x -- The input of real images for domain X.
            input_y -- The input of real images for domain Y.

        Returns:
            The two discriminator objects.
        '''

        if self.config[cfg.CLASS_MODE] == cfg.CLASS_LOSS:
            if self.config[cfg.TRAINING_CONFIG][cfg.CLASS_LOSS_TYPE] == cfg.NPLUSONE:
                target_channels = self.config[cfg.CLASS_COUNT] + 1
            else:
                target_channels = self.config[cfg.CLASS_COUNT] * 2
        else:
            target_channels = 1

        discriminator_x = self.DiscriminatorClass(
            self.DISCRIMINATOR_X_NAME, self.config[cfg.INSTANCE_NORM_AFFINE],
            self.config[cfg.TRAINING_CONFIG][cfg.DISCRIMINATOR_X_INITIAL_FILTER_COUNT],
            target_channels = target_channels
        )
        discriminator_y = self.DiscriminatorClass(
            self.DISCRIMINATOR_Y_NAME, self.config[cfg.INSTANCE_NORM_AFFINE],
            self.config[cfg.TRAINING_CONFIG][cfg.DISCRIMINATOR_Y_INITIAL_FILTER_COUNT],
            target_channels = target_channels
        )


        #Create discriminator graph version with input data as input.
        discriminator_x.from_input = TupleWrapper(discriminator_x.create(input_x, False))
        discriminator_y.from_input = TupleWrapper(discriminator_y.create(input_y, False))

        #Create discriminator graph version with mapping as input.
        discriminator_x.from_mapping = TupleWrapper(discriminator_x.create(self.mapping_yx.from_input.all, True))
        discriminator_y.from_mapping = TupleWrapper(discriminator_y.create(self.mapping_xy.from_input.all, True))

        #Create discriminator graph version with the image buffer as input
        discriminator_x.from_image_buffer = TupleWrapper(discriminator_x.create(self.image_buffer_x, True))
        discriminator_y.from_image_buffer = TupleWrapper(discriminator_y.create(self.image_buffer_y, True))

        return discriminator_x, discriminator_y

    def create_cycle_gan(self, *inputs):
        '''
        Initializes the CycleGAN's graph structure (the two mappings and discriminators).

        Arguments:
            inputs  -- A list of inputs. Must be length 2 if the CycleGAN's mode is set such that both mappings have been built, otherwise must be length 1.
        '''

        if self.mode == Mode.RUN_XY:
            input_x = inputs[0]
            input_y = None
        elif self.mode == Mode.RUN_YX:
            input_x = None
            input_y = inputs[0]
        else:
            input_x = inputs[0]
            input_y = inputs[1]

        self.mapping_xy, self.mapping_yx = self.create_mappings(input_x, input_y)

        if self.mode == Mode.TRAIN:
            self.image_buffer_x = ops.tensor_buffer(self.mapping_yx.from_input.all, self.config[cfg.TRAINING_CONFIG][cfg.IMAGE_BUFFER_SIZE])
            self.image_buffer_y = ops.tensor_buffer(self.mapping_xy.from_input.all, self.config[cfg.TRAINING_CONFIG][cfg.IMAGE_BUFFER_SIZE])

            self.discriminator_x, self.discriminator_y = self.create_discriminators(input_x, input_y)

    def get_discriminator_x_variables(self):
        '''
        Retrieves the list of variables for the discriminator of domain X.

        Returns:
            A list of variables, or an empty list if the CycleGAN's mode is set such that the discriminator has not been built.
        '''

        if self.mode == Mode.TRAIN:
            return self.discriminator_x.get_variables()

        return list()

    def get_discriminator_y_variables(self):
        '''
        Retrieves the list of variables for the discriminator of domain Y.

        Returns:
            A list of variables, or an empty list if the CycleGAN's mode is set such that the discriminator has not been built.
        '''

        if self.mode == Mode.TRAIN:
            return self.discriminator_y.get_variables()

        return list()

    def get_mapping_variables(self):
        '''
        Retrieves the list of variables for all built mappings.

        Returns:
            A list of variables.
        '''

        variables = list()

        if self.mode != Mode.RUN_YX:
            variables += self.mapping_xy.get_variables()

        if self.mode != Mode.RUN_XY:
            variables += self.mapping_yx.get_variables()

        return variables


def get_vector_decorated_mapping(concat, passthrough):
    '''
    Creates a class-like lambda object that is vector-decorated mapping with the given concat and passthrough values.

    Arguments:
        concat      -- The concat list for the decorator.
        passthrough -- The passthrough list for the decorator.

    Returns:
        The vector-decorator constructor function.
    '''

    return lambda *args, **kws: VectorDecorator(Mapping, concat, passthrough, *args, **kws)

def get_vector_decorated_discriminator(concat, passthrough):
    '''
    Creates a class-like lambda object that is vector-decorated discriminator with the given concat and passthrough values.

    Arguments:
        concat      -- The concat list for the decorator.
        passthrough -- The passthrough list for the decorator.

    Returns:
        The vector-decorator constructor function.
    '''

    return lambda *args, **kws: VectorDecorator(Discriminator, concat, passthrough, *args, **kws)

def get_info_gan_discriminator(output_size):
    '''
    Creates a class-like lambda object that is a infoGAN discriminator.

    Arguments:
        output_size     -- The size of the infoGAN discriminator's secondary output head.

    Returns:
        The infoGAN discriminator function.
    '''

    VectorDecoratedDiscriminator = get_vector_decorated_discriminator([False], [True])
    return lambda *args, **kws: InfoGANDecorator(VectorDecoratedDiscriminator, output_size, *args, **kws)
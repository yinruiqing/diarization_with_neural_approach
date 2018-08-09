import io
import yaml
from os.path import dirname, expanduser
import numpy as np

from docopt import docopt



from pyannote.database.util import get_annotated
from pyannote.database import get_protocol
from pyannote.database import get_unique_identifier

from pyannote.parser import MDTMParser

from pyannote.audio.util import mkdir_p

from pyannote.audio.features.utils import Precomputed
import h5py



from pyannote.audio.generators.periodic import PeriodicFeaturesMixin
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.core import Segment, Timeline, Annotation
from pyannote.generators.fragment import SlidingSegments
from pyannote.generators.batch import FileBasedBatchGenerator
import numpy as np

import random 

from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.audio.callback import LoggingCallback
from pyannote.audio.keras_utils import CUSTOM_OBJECTS

class SequenceLabeling(PeriodicFeaturesMixin, FileBasedBatchGenerator):
    """Sequence labeling
    Parameters
    ----------
    model : keras.Model
        Pre-trained sequence labeling model.
    feature_extraction : callable
        Feature extractor
    duration : float
        Subsequence duration, in seconds.
    step : float, optional
        Subsequence step, in seconds. Defaults to 50% of `duration`.
    batch_size : int, optional
        Defaults to 32.
    gpu : boolean, optional
        Run on GPU. Only works witht pytorch backend.
    Usage
    -----
    >>> model = keras.models.load_model(...)
    >>> feature_extraction = YaafeMFCC(...)
    >>> sequence_labeling = SequenceLabeling(model, feature_extraction, duration)
    >>> sequence_labeling.apply(current_file)
    """

    def __init__(self, model, duration,
                 step=None, batch_size=32, source='audio', gpu=False):

        self.model = model.cuda() if gpu else model
        self.duration = duration
        self.batch_size = batch_size
        self.gpu = gpu

        generator = SlidingSegments(duration=duration, step=step, source=source)
        self.step = generator.step if step is None else step

        super(SequenceLabeling, self).__init__(
            generator, batch_size=self.batch_size)

    @property
    def dimension(self):
        return self.model.output_shape[-1]

    def signature(self):
        return {'type': 'ndarray'}
    
    def preprocess(self, current_file, identifier=None):
        """Pre-compute file-wise X"""
        
        
        if not hasattr(self, 'preprocessed_'):
            self.preprocessed_ = {}
            self.preprocessed_['X'] = {}
        if not hasattr(self, 'sliding_window'):
            self.sliding_window = current_file['features'].sliding_window
            
        self.preprocessed_['X'][identifier] = current_file['features']


        return current_file



    def postprocess_ndarray(self, X):
        """Label sequences
        Parameter
        ---------
        X : (batch_size, n_samples, n_features) numpy array
            Batch of input sequences
        Returns
        -------
        prediction : (batch_size, n_samples, dimension) numpy array
            Batch of sequence labelings.
        """
        
        return self.model.predict(X)

    def apply(self, current_file):
        """Compute predictions on a sliding window
        Parameter
        ---------
        current_file : dict
        Returns
        -------
        predictions : SlidingWindowFeature
        """

        # frame and sub-sequence sliding windows
        

        batches = [batch for batch in self.from_file(current_file,
                                                     incomplete=True)]
        frames = self.sliding_window
        if not batches:
            data = np.zeros((0, self.dimension), dtype=np.float32)
            return SlidingWindowFeature(data, frames)

        fX = np.vstack(batches)

        subsequences = SlidingWindow(duration=self.duration, step=self.step)

        # get total number of frames
        identifier = get_unique_identifier(current_file)
        n_frames = self.preprocessed_['X'][identifier].data.shape[0]

        # data[i] is the sum of all predictions for frame #i
        data = np.zeros((n_frames, self.dimension), dtype=np.float32)

        # k[i] is the number of sequences that overlap with frame #i
        k = np.zeros((n_frames, 1), dtype=np.int8)

        for subsequence, fX_ in zip(subsequences, fX):

            # indices of frames overlapped by subsequence
            indices = frames.crop(subsequence,
                                  mode='center',
                                  fixed=self.duration)

            # accumulate the outputs
            data[indices] += fX_

            # keep track of the number of overlapping sequence
            # TODO - use smarter weights (e.g. Hamming window)
            k[indices] += 1

        # compute average embedding of each frame
        data = data / np.maximum(k, 1)
        data = np.argmax(data, axis=1)
        return SlidingWindowFeature(data, frames)

    @classmethod
    def train(cls, input_shape, design_model, generator, steps_per_epoch,
              epochs, loss='categorical_crossentropy', optimizer='rmsprop',
              log_dir=None):
        """Train the model
        Parameters
        ----------
        input_shape : (n_frames, n_features) tuple
            Shape of input sequence
        design_model : function or callable
            This function should take input_shape as input and return a Keras
            model that takes a sequence as input, and returns the labeling as
            output.
        generator : iterable
            The output of the generator must be a tuple (inputs, targets) or a
            tuple (inputs, targets, sample_weights). All arrays should contain
            the same number of samples. The generator is expected to loop over
            its data indefinitely. An epoch finishes when `steps_per_epoch`
            samples have been seen by the model.
        steps_per_epoch : int
            Number of batches to process before going to the next epoch.
        epochs : int
            Total number of iterations on the data
        optimizer: str, optional
            Keras optimizer. Defaults to 'rmsprop'.
        log_dir: str, optional
            When provided, log status after each epoch into this directory.
            This will create several files, including loss plots and weights
            files.
        See also
        --------
        keras.engine.training.Model.fit_generator
        """

        callbacks = []

        if log_dir is not None:
            log = [('train', 'loss'), ('train', 'accuracy')]
            callback = LoggingCallback(log_dir, log=log)
            callbacks.append(callback)

        # in case the {generator | optimizer} define their own
        # callbacks, append them as well. this might be useful.
        for stuff in [generator, optimizer]:
            if hasattr(stuff, 'callbacks'):
                callbacks.extend(stuff.callbacks())

        model = design_model(input_shape)
        model.compile(optimizer=optimizer,
                               loss=loss,
                               metrics=['accuracy'])

        model.fit_generator(
            generator, steps_per_epoch, epochs=epochs,
            verbose=1, callbacks=callbacks)
        return model


class RealignmentBatchGenerator(PeriodicFeaturesMixin,
                                    FileBasedBatchGenerator):

    """(X_batch, y_batch) batch generator
    Yields batches made of subsequences obtained using a sliding
    window over the audio files.
    Parameters
    ----------
    duration: float, optional
        yield segments of length `duration`
        Defaults to 3.2s.
    step: float, optional
        step of sliding window (in seconds).
        Defaults to 0.8s.
    batch_size: int, optional
        Size of batch
        Defaults to 32
    Returns
    -------
    X_batch : (batch_size, n_samples, n_features) numpy array
        Batch of feature sequences
    y_batch : (batch_size, n_samples) numpy array
        Batch of corresponding label sequences
    Usage
    -----
    >>> batch_generator = RealignmentBatchGenerator(feature_extractor)
    >>> for X_batch, y_batch in batch_generator.from_file(current_file):
    ...     # do something with
    """

    def __init__(self, duration=3.2, step=0.8, batch_size=32, source='annotated'):

        self.duration = duration
        self.step = step
        self.source= source
        segment_generator = SlidingSegments(duration=duration,
                                            step=step,
                                            source=source)
        super(RealignmentBatchGenerator, self).__init__(
            segment_generator, batch_size=batch_size)

    def signature(self):
        #n_samples = self.feature_extractor.sliding_window().samples(
        #            self.duration, mode='center')
        #dimension = self.feature_extractor.dimension()
        #self.shape = (n_samples, dimension)
        #shape = self.shape

        return [
            #{'type': 'ndarray', 'shape': shape},
            #{'type': 'ndarray', 'shape': (shape[0], 2)}
            {'type': 'ndarray'},
            {'type': 'ndarray'}
        ]

    def preprocess(self, current_file, identifier=None):
        """Pre-compute file-wise X and y"""
        
        
        if not hasattr(self, 'preprocessed_'):
            self.preprocessed_ = {}
            self.preprocessed_['X'] = {}
            self.preprocessed_['y'] = {}
            
            
        self.preprocessed_['X'][identifier] = current_file['features']

        # if labels have already been extracted, do nothing
        if identifier in self.preprocessed_.setdefault('y', {}):
            return current_file

        # get features as pyannote.core.SlidingWindowFeature instance
        X = self.preprocessed_['X'][identifier]
        sw = X.sliding_window
        n_samples = X.getNumber()
        #self.shape = (n_samples, dimension)
        
        annotated = get_annotated(current_file)
        annotation = current_file['annotation']
        prediction = current_file['prediction']
        
        if not hasattr(self, 'input_shape'):
            self.input_shape = (sw.samples(self.duration, mode='center'), X.data.shape[1])
        
        
        if self.source == 'annotation':
        	n_classes = len(prediction.labels())
        	self.n_classes = n_classes
        	y = np.zeros((n_samples + 4, n_classes), dtype=np.int8)
        	label_map = {label: idx for idx, label in enumerate(prediction.labels())}
        else:
        	n_classes = len(prediction.labels())+1
        	self.n_classes = n_classes
        	y = np.zeros((n_samples + 4, n_classes), dtype=np.int8)
        	label_map = {label: idx+1 for idx, label in enumerate(prediction.labels())}

        for segment, _, label in prediction.itertracks(label=True):
            indices = sw.crop(segment, mode='loose')
            y[indices,label_map[label]] = 1

        y = SlidingWindowFeature(y[:-1], sw)
        self.preprocessed_['y'][identifier] = y

        return current_file

    
    # defaults to extracting frames centered on segment
    def process_segment(self, segment, signature=None, identifier=None):
        """Extract X and y subsequences"""

        X = self.periodic_process_segment(
            segment, signature=signature, identifier=identifier)

        duration = signature.get('duration', None)

        y = self.preprocessed_['y'][identifier].crop(
            segment, mode='center', fixed=duration)
        return [X, y]

class Realignment(object):
    def __init__(self, feature_precomputed, config_yml, num_epoch=5, 
                subset='development', source = 'annotated', 
                models_dir = None):
        self.feature_precomputed = feature_precomputed
        with open(config_yml, 'r') as fp:
            self.config_ = yaml.load(fp)
        self.num_epoch = num_epoch
        self.source = source
        self.models_dir = models_dir


    def train(self, current_file, batch_size=32):
        def generator(xs, ys, batch_size, shuffle=True):
            length = len(xs)
            idxs = list(range(length))
            if shuffle:
                random.shuffle(idxs)
            while True:
                tmp = []
                for i in idxs:
                    tmp.append(i)
                    if len(tmp) == batch_size:
                        xbatch = np.vstack([xs[i] for i in tmp])
                        ybatch = np.vstack([ys[i] for i in tmp])
                        tmp = []
                        yield xbatch, ybatch
        
        duration = self.config_['sequences']['duration']
        step = self.config_['sequences']['step']
        
        current_file['features'] = self.feature_precomputed(current_file)
        realignment_generator = RealignmentBatchGenerator(
                             duration=duration, step=step,
                             batch_size=1, source = self.source)
        
        bg = realignment_generator.from_file(current_file)
        xys = [(x,y) for x,y in bg]
        xs = [x for x,_ in xys]
        ys = [y for _,y in xys]
        
        input_shape = realignment_generator.input_shape
        n_classes = realignment_generator.n_classes
        # architecture
        architecture_name = self.config_['architecture']['name']
        models = __import__('pyannote.audio.labeling.models',
                            fromlist=[architecture_name])
        Architecture = getattr(models, architecture_name)
        params = self.config_['architecture'].get('params', {})
        params['n_classes'] = n_classes
        self.architecture_ = Architecture(**params)
        
        train_total = sum([end-start for start, end in current_file['annotated']])
        steps_per_epoch = int(np.ceil((train_total / step) / batch_size))
        
        if self.models_dir is None:
            return SequenceLabeling.train(
                            input_shape, self.architecture_, generator(xs, ys, batch_size, shuffle=True), 
                            steps_per_epoch, self.num_epoch,
                            optimizer=SSMORMS3(),
                            log_dir=None)
        else: 
            return SequenceLabeling.train(
                                input_shape, self.architecture_, generator(xs, ys, batch_size, shuffle=True), 
                                steps_per_epoch, self.num_epoch,
                                optimizer=SSMORMS3(),
                                log_dir=self.models_dir+get_unique_identifier(current_file))
    
        
    def load_model(self, train_dir, epoch, compile=True):
            import keras.models
            if train_dir is None:
                train_dir = self.models_dir
            WEIGHTS_H5 = '{train_dir}/weights/{epoch:04d}.h5'
            weights_h5 = WEIGHTS_H5.format(train_dir=train_dir, epoch=epoch)
                
            model = keras.models.load_model(weights_h5,
                custom_objects=CUSTOM_OBJECTS, compile=compile)
            model.epoch = epoch
            return model
    
    def apply(self, current_file, model=None):
        duration = self.config_['sequences']['duration']
        step = self.config_['sequences']['step']
        if model is None:
            model = self.train(current_file)
        sequence_labeling = SequenceLabeling(
                            model, duration= duration,
                            step=step)
    
        prediction =  sequence_labeling.apply(current_file)
        boundries = [0]  + list(np.where(prediction.data[1:] - prediction.data[:-1] != 0)[0]+1)
        pred_annotation = Annotation()
        for start_ind, end_ind in zip(boundries[:-1],boundries[1:]):
            start, end = prediction.sliding_window[start_ind].middle, prediction.sliding_window[end_ind].middle
            if self.source == 'annotated' and prediction.data[start_ind] == 0:
            	continue
            pred_annotation[Segment(start, end)] = prediction.data[start_ind]
            
        return pred_annotation

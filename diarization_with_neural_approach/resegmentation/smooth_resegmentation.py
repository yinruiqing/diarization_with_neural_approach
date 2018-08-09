
"""
Re-segmentation

Usage:
    smooth_resegmentation  <config_yml> <feature> <feature_pre> <num_epoch> <database.task.protocol> <subset> <models_dir> <diarization.mdtm> <output_filename>

Parameters:
<config_yml>                      Config file for feature extraction.
<feature>                         Feature used to do the re-segmentation
                                  [default: mfcc]
<feature_pre>                     Path for precomputed feature
<num_epoch>                       The number of training epoches
<database.task.protocol>          Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
<subset>                          Set subset (train|developement|test).
<models_dir>                      The directory path to stored the trained models
<diarization.mdtm>                The prediction before re-segmention
<output_filename>                 The file to store the evaluations during the re-segmentation
"""

import sys
sys.path.append("./")
sys.path.append("../")
from resegment import SequenceLabeling, RealignmentBatchGenerator, Realignment

from docopt import docopt
import io
import yaml
from os.path import dirname, expanduser
import numpy as np

from pyannote.database.util import FileFinder


from pyannote.database import get_protocol
from pyannote.database import get_unique_identifier


from pyannote.audio.util import mkdir_p

from pyannote.audio.features.utils import Precomputed
import h5py

from pyannote.metrics.diarization import DiarizationPurityCoverageFMeasure
from pyannote.parser import MDTMParser

from pyannote.core import Segment, Timeline, Annotation
from pyannote.generators.fragment import SlidingSegments
from pyannote.database.util import get_annotated
import random 

def get_ders(l):
    ders = {}
    for i in range(len(l)):
        report = l[i].report()
        ders[i] = report['diarization error rate'].to_dict()['%']
    return ders

def getEpochNums(n,window=3):
    l = [n+i-window+1 for i in range(window)]
    return [i for i in l if i >=0]

if __name__ == "__main__":
    arguments = docopt(__doc__, version='re-segmentation')

    protocol = get_protocol(arguments['<database.task.protocol>'],
                            preprocessors={'audio': FileFinder()})
    subset = arguments['<subset>']

    diarization_mdtm = arguments['<diarization.mdtm>']
    parser = MDTMParser()
    annotations = parser.read(diarization_mdtm)

    diarization_res = {}
    for uri in annotations.uris:
        if uri not in diarization_res:
            diarization_res[uri] = Annotation(uri=uri)
        diarization_res[uri].update(annotations(uri=uri, modality="speaker"))

    from pyannote.metrics.diarization import GreedyDiarizationErrorRate
    metric1 = GreedyDiarizationErrorRate(parallel=False)
    metric2 = GreedyDiarizationErrorRate(parallel=False,collar=0.500, skip_overlap=True)
    metric3 = GreedyDiarizationErrorRate(parallel=False,collar=0.500, skip_overlap=False)

    from optimize_cluster import speaker_diarization 
    from pyannote.audio.features import Precomputed

    file_list = []
    for current_file in getattr(protocol, subset)():
        uri = get_unique_identifier(current_file).split('/')[1]
        hypothesis = diarization_res[uri]
        reference = current_file['annotation']
        current_file['prediction'] = hypothesis
        file_list.append(current_file)
        uem = get_annotated(current_file)
        metric1(reference, hypothesis, uem=uem)
        metric2(reference, hypothesis, uem=uem)
        metric3(reference, hypothesis, uem=uem)

    print(abs(metric1))
    print(abs(metric2))
    print(abs(metric3))


    config_yml = arguments['<config_yml>']
    models_dir = arguments['<models_dir>']
    num_epoch = int(arguments['<num_epoch>'])
    embedding_precomputed = '/vol/work1/yin/embedding/20180124'
    

    feature_precomputed = arguments['<feature_pre>']
    
    ra = Realignment(feature_precomputed, config_yml, num_epoch = num_epoch, 
                    source = 'annotated', models_dir = models_dir)
    for current_file in file_list:
        ra.train(current_file)

    res_all = [GreedyDiarizationErrorRate(parallel=False) for i in range(num_epoch)]

    for current_file in file_list:
        train_dir = models_dir+get_unique_identifier(current_file)
        for epoch in range(num_epoch):
            models = [ra.load_model(train_dir, e) for e in getEpochNums(epoch)]
            pred = ra.applyModels( current_file, models)
            timeline = current_file['prediction'].get_timeline()
            pred = pred.crop(timeline)
            reference = current_file['annotation']
            uem = get_annotated(current_file)
            res_all[epoch](reference, pred, uem=uem)
    results = get_ders(res_all)
    import simplejson as json
    with open(arguments['<output_filename>']+ subset +'_'+str(num_epoch)+'.json', 'w') as f:
        json.dump(results,f)

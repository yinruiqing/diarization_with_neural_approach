# Speaker diarization with neural approach
Code for [**Neural speech turn segmentation and affinity propagation for speaker diarization**](https://github.com/yinruiqing/change_detection/blob/master/doc/change-detection.pdf)

**Attention:** The pipeline has been merged in [`pyannote.audio`](https://github.com/pyannote/pyannote-audio). The codes here are just for reproducing the results in the paper.

## Citation

```
@inproceedings{yin2018neural,
  title={{Neural speech turn segmentation and affinity propagation for speaker diarization}},
  author={ Ruiqing Yin and Herv\'e Bredin and Claude Barras},
  year={2018},
  Booktitle={{Interspeech 2018, 19th Annual Conference of the International Speech Communication Association}}
}
```

## Installation

**Foreword:** The code is based on [`pyannote`](https://github.com/pyannote). You can also find a similar `Readme` file for [`TristouNet `](https://github.com/hbredin/TristouNet).

```bash
$ conda create --name neural-diarization python=3.6 anaconda
$ source activate neural-diarization
$ conda install gcc
$ conda install -c yaafe yaafe=0.65
$ pip install "pyannote.audio==0.3"
$ pip install pyannote.db.etape
$ pip install git+https://github.com/hyperopt/hyperopt.git
```

## Usage
### Speech activity detection
Please follow the [`documentation `](https://github.com/pyannote/pyannote-audio/blob/0.3/tutorials/speech-activity-detection) in [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) to train and select a SAD model. Then store the SAD scores for optimization step.
### Speaker change detection
Please follow the [`documentation `](https://github.com/pyannote/pyannote-audio/blob/0.3/tutorials/change-detection) in [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) to train and select a SCD model. Then store the SCD scores for optimization step.
### Embedding training and extraction
Please follow the [`documentation `](https://github.com/pyannote/pyannote-audio/tree/master/tutorials/speaker-embedding) in [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) to extract embeddings for optimization step.
### Optimization
We use [`Hyperopt `]http://hyperopt.github.io/hyperopt/)  to do the global optimization. The objective function is in 
Then you can use the following code to evaluate and generate the mdtm file.
```python
args={'cls__damping': 0.8543270898940996,
 'cls__metric': 'cosine',
 'cls__preference': -7.027266273489084,
 'emb__internal': False,
 'sad__dimension': 1,
 'sad__offset': 0.7057908535808806,
 'sad__onset': 0.8928696298724925,
 'scd__alpha': 0.00020692277092768802,
 'scd__dimension': 1,
 'scd__min_duration': 1.5350520647259391}


import sys
sys.path.append("../")
from pyannote.database import get_protocol, get_annotated, FileFinder
protocol = get_protocol('Etape.SpeakerDiarization.TV',
                        preprocessors={'audio': FileFinder()})

from pyannote.metrics.diarization import GreedyDiarizationErrorRate
metric1 = GreedyDiarizationErrorRate(parallel=False)
metric2 = GreedyDiarizationErrorRate(parallel=False,collar=0.500, skip_overlap=True)
metric3 = GreedyDiarizationErrorRate(parallel=False,collar=0.500, skip_overlap=False)

from optimize_cluster import speaker_diarization 
from pyannote.audio.features import Precomputed

feature_extraction = Precomputed('path/for/precomputed/mfcc')
sad_pre = 'path/for/precomputed/sad/score'
scd_pre = 'path/for/precomputed/scd/score'
emb_pre = 'path/for/embedding'

args['cls__damping'] = float(args['cls__damping'])
args['cls__preference'] = float(args['cls__preference'])

pipeline = speaker_diarization.SpeakerDiarizationPre(feature_extraction, sad_pre, scd_pre, emb_pre, **args)

file_list = []
for current_file in protocol.development():
    hypothesis = pipeline(current_file, annotated=True)
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

from pyannote.parser import MDTMParser
path = 'path.mdtm'
writer = MDTMParser()
with io.open(path, mode='w') as gp:
    for current_file in file_list:
        writer.write(current_file['prediction'],
                    f=gp, uri=current_file['uri'], modality='speaker')
```
Output
```bash
0.173114946610612
0.09515458406500096
0.1198648679786544
```
### Re-segmentation

Re-segmentation
Please use the following instructions to do the re-segmentation and generate the mdtm file.
```bash
python smooth_resegmentation.py -h

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
```
```python
from resegment import Realignment
ra =  Realignment(feature_precomputed, config_yml, 
                  num_epoch = 2, source = 'annotated', models_dir = 'tmp/')
epoch = epoch_number
models_dir = '/models/dir/'
path = 'orignal.mdtm'
writer = MDTMParser()
with io.open(path, mode='w') as gp:
    for current_file in file_list:
        train_dir = models_dir+get_unique_identifier(current_file)
        current_file['features'] = feature_precomputed(current_file)
        model = ra.load_model(train_dir, epoch)
        pred = ra.apply( current_file, model)
        timeline = current_file['prediction'].get_timeline()
        pred = pred.crop(timeline)
        writer.write(pred,
                    f=gp, uri=current_file['uri'], modality='speaker')
```



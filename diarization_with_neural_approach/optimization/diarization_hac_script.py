# --- experiment unique identifier ------------------
xp_name = 'diarization_hac_script'

# --- hyper-parameters search space -----------------
from hyperopt import hp
xp_space = {'sad__onset': hp.uniform('sad__onset', 0.6, 1.0),
            'sad__offset': hp.uniform('sad__offset', 0.6, 1.0),
            'sad__dimension': 1,
            'scd__alpha': hp.uniform('scd__alpha', 0., 0.4),
            'scd__min_duration': hp.uniform('scd__min_duration', 0.5, 5.),
            'scd__dimension': 1,
            'emb__internal': False,
            'cls__method':'complete',  # average, single
            'cls__threshold': hp.uniform('cls__threshold', 0.,3.1),
            'cls__metric': 'angular'}


# --- objective function ----------------------------
def xp_objective(args, **kwargs):
    import sys
    sys.path.append("/people/yin/projects/")
    from pyannote.database import get_protocol, get_annotated, FileFinder
    protocol = get_protocol('Etape.SpeakerDiarization.TV',
                            preprocessors={'audio': FileFinder()})

    from pyannote.metrics.diarization import GreedyDiarizationErrorRate
    metric = GreedyDiarizationErrorRate()

    from optimize_cluster import speaker_diarization 
    from pyannote.audio.features import Precomputed

    feature_extraction = Precomputed('/vol/work1/bredin/feature_extraction/mfcc')
    sad_pre = '/vol/work1/bredin/speech_activity_detection/shallow/train/REPERE.SpeakerDiarization.All.train/tune/Etape.SpeakerDiarization.TV.development/apply'
    scd_pre = '/vol/work1/bredin/speaker_change_detection/paper/train/REPERE.SpeakerDiarization.All.train/tune/Etape.SpeakerDiarization.Debug.development/apply'
    emb_pre = '/vol/work1/yin/embedding/20180124'

    args['cls__threshold'] = float(args['cls__threshold'])
    
    pipeline = speaker_diarization.SpeakerDiarizationHACPre(feature_extraction, sad_pre, scd_pre, emb_pre, **args)

    try:
        for current_file in protocol.development():
            hypothesis = pipeline(current_file, annotated=True)
            if hypothesis is None:
                return 100
            reference = current_file['annotation']
            uem = get_annotated(current_file)
            metric(reference, hypothesis, uem=uem)
    except MemoryError as error:
        return 100

    return abs(metric)
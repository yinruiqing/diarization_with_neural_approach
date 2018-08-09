import numpy as np
from pyannote.audio.keras_utils import load_model
from pyannote.audio.signal import Binarize, Peak
from pyannote.audio.features import Precomputed
import my_cluster
from pyannote.core import Annotation
from pyannote.audio.embedding.utils import l2_normalize
from pyannote.database import get_annotated



class SpeakerDiarizationPre(object):
    '''Speaker diarization with affinity propagation'''

    def __init__(self, feature_extraction, sad__pre, scd__pre, emb__pre,
                 sad__onset=0.7, sad__offset=0.7, sad__dimension=1,
                 scd__alpha=0.5, scd__min_duration=1., scd__dimension=1,
                 emb__internal=False,
                 cls__damping=0.8, cls__preference=-20,
                 cls__metric='cosine'):

        super(SpeakerDiarizationPre, self).__init__()

        self.feature_extraction = feature_extraction

        # speech activity detection hyper-parameters
        self.sad__onset = sad__onset
        self.sad__offset = sad__offset
        self.sad__dimension = sad__dimension

        # speaker change detection hyper-parameters
        self.scd__alpha = scd__alpha
        self.scd__min_duration = scd__min_duration
        self.scd__dimension = scd__dimension

        # embedding hyper-parameters
        self.emb__internal = emb__internal

        # clustering hyper-parameters
        self.cls__damping = cls__damping
        self.cls__preference = cls__preference
        self.cls__metric = cls__metric

        step = self.feature_extraction.sliding_window().step

        # initialize speech activity detection module
        self.sad_ = Precomputed(sad__pre)
        self.sad_binarize_ = Binarize(onset=self.sad__onset,
                                      offset=self.sad__offset)

        # initialize speaker change detection module
        self.scd_ = Precomputed(scd__pre)
        self.scd_peak_ = Peak(alpha=self.scd__alpha,
                              min_duration=self.scd__min_duration,
                              percentile=False)

        # initialize speech turn embedding module
        self.emb_ = Precomputed(emb__pre)

        # initialize clustering module
        self.cls_ = my_cluster.ClusteringAP(metric=self.cls__metric,
                                        damping=self.cls__damping,
                                        preference=self.cls__preference)


    def __call__(self, current_file, annotated=False):

        # speech activity detection
        soft_sad = self.sad_(current_file)
        hard_sad = self.sad_binarize_.apply(
            soft_sad, dimension=self.sad__dimension)

        # speaker change detection
        soft_scd = self.scd_(current_file)
        hard_scd = self.scd_peak_.apply(
            soft_scd, dimension=self.scd__dimension)

        # speech turns
        speech_turns = hard_scd.crop(hard_sad)

        if annotated:
            speech_turns = speech_turns.crop(
                get_annotated(current_file))

        # remove small speech turns
        emb = self.emb_(current_file)
        speech_turns = [speech_turn for speech_turn in speech_turns if len(emb.crop(speech_turn, mode='loose')) > 0]

        # speech turns embedding
        to_stack = [
            np.sum(emb.crop(speech_turn, mode='loose'), axis=0)
            for speech_turn in speech_turns]
        if len(to_stack) < 1:
            return None
        fX = l2_normalize(np.vstack(to_stack))

        # speech turn clustering
        cluster_labels = self.cls_.apply(fX)

        # build hypothesis from clustering results
        hypothesis = Annotation(uri=current_file['uri'])
        for speech_turn, label in zip(speech_turns, cluster_labels):
            hypothesis[speech_turn] = label
        return hypothesis


class SpeakerDiarizationOracleSegAP(object):
    '''Speaker diarization with oracle segmentation and affinity propagation'''

    def __init__(self, feature_extraction, emb__pre,
                 emb__internal=False,
                 cls__damping=0.8, cls__preference=-20,
                 cls__metric='cosine'):

        super(SpeakerDiarizationOracleSegAP, self).__init__()

        self.feature_extraction = feature_extraction

        # embedding hyper-parameters
        self.emb__internal = emb__internal

        # clustering hyper-parameters
        self.cls__damping = cls__damping
        self.cls__preference = cls__preference
        self.cls__metric = cls__metric

        step = self.feature_extraction.sliding_window().step

        # initialize speech turn embedding module
        self.emb_ = Precomputed(emb__pre)

        # initialize clustering module
        self.cls_ = my_cluster.ClusteringAP(metric=self.cls__metric,
                                        damping=self.cls__damping,
                                        preference=self.cls__preference)


    def __call__(self, current_file, annotated=False):


        # speech turns
        speech_turns = current_file['annotation'].get_timeline()

        if annotated:
            speech_turns = speech_turns.crop(
                get_annotated(current_file))

        # remove small speech turns
        emb = self.emb_(current_file)
        speech_turns = [speech_turn for speech_turn in speech_turns if len(emb.crop(speech_turn, mode='loose')) > 0]

        # speech turns embedding
        to_stack = [
            np.sum(emb.crop(speech_turn, mode='loose'), axis=0)
            for speech_turn in speech_turns]
        if len(to_stack) < 1:
            return None
        fX = l2_normalize(np.vstack(to_stack))

        # speech turn clustering
        cluster_labels = self.cls_.apply(fX)

        # build hypothesis from clustering results
        hypothesis = Annotation(uri=current_file['uri'])
        for speech_turn, label in zip(speech_turns, cluster_labels):
            hypothesis[speech_turn] = label
        return hypothesis


class SpeakerDiarizationHACPre(object):
    '''Speaker diarization with hierarchical agglomerative clustering'''

    def __init__(self, feature_extraction, sad__pre, scd__pre, emb__pre,
                 sad__onset=0.7, sad__offset=0.7, sad__dimension=1,
                 scd__alpha=0.5, scd__min_duration=1., scd__dimension=1,
                 emb__internal=False,
                 cls__method='average', cls__threshold=5,
                 cls__metric='cosine'):

        super(SpeakerDiarizationHACPre, self).__init__()

        self.feature_extraction = feature_extraction

        # speech activity detection hyper-parameters
        self.sad__onset = sad__onset
        self.sad__offset = sad__offset
        self.sad__dimension = sad__dimension

        # speaker change detection hyper-parameters
        self.scd__alpha = scd__alpha
        self.scd__min_duration = scd__min_duration
        self.scd__dimension = scd__dimension

        # embedding hyper-parameters
        self.emb__internal = emb__internal

        # clustering hyper-parameters
        self.cls__method = cls__method
        self.cls__threshold = cls__threshold
        self.cls__metric = cls__metric

        step = self.feature_extraction.sliding_window().step

        # initialize speech activity detection module
        self.sad_ = Precomputed(sad__pre)
        self.sad_binarize_ = Binarize(onset=self.sad__onset,
                                      offset=self.sad__offset)

        # initialize speaker change detection module
        self.scd_ = Precomputed(scd__pre)
        self.scd_peak_ = Peak(alpha=self.scd__alpha,
                              min_duration=self.scd__min_duration,
                              percentile=False)

        # initialize speech turn embedding module
        self.emb_ = Precomputed(emb__pre)

        # initialize clustering module
        self.cls_ = my_cluster.ClusteringHAC(metric=self.cls__metric,
                                        method=self.cls__method,
                                        threshold=self.cls__threshold)


    def __call__(self, current_file, annotated=False):

        # speech activity detection
        soft_sad = self.sad_(current_file)
        hard_sad = self.sad_binarize_.apply(
            soft_sad, dimension=self.sad__dimension)

        # speaker change detection
        soft_scd = self.scd_(current_file)
        hard_scd = self.scd_peak_.apply(
            soft_scd, dimension=self.scd__dimension)

        # speech turns
        speech_turns = hard_scd.crop(hard_sad)

        if annotated:
            speech_turns = speech_turns.crop(
                get_annotated(current_file))

        # remove small speech turns
        emb = self.emb_(current_file)
        speech_turns = [speech_turn for speech_turn in speech_turns if len(emb.crop(speech_turn, mode='loose')) > 0]

        # speech turns embedding
        to_stack = [
            np.sum(emb.crop(speech_turn, mode='loose'), axis=0)
            for speech_turn in speech_turns]
        if len(to_stack) < 1:
            return None
        fX = l2_normalize(np.vstack(to_stack))

        # speech turn clustering
        cluster_labels = self.cls_.apply(fX)

        # build hypothesis from clustering results
        hypothesis = Annotation(uri=current_file['uri'])
        for speech_turn, label in zip(speech_turns, cluster_labels):
            hypothesis[speech_turn] = label
        return hypothesis



class SpeakerDiarizationPreStages(object):

    def __init__(self, feature_extraction, sad__pre, scd__pre, emb__pre,
                 sad__onset=0.7, sad__offset=0.7, sad__dimension=1,
                 scd__alpha=0.5, scd__min_duration=1., scd__dimension=1,
                 emb__internal=False,
                 cls__damping=0.8, cls__preference=-20,
                 cls__metric='cosine'):

        super(SpeakerDiarizationPreStages, self).__init__()

        self.feature_extraction = feature_extraction

        # speech activity detection hyper-parameters
        self.sad__onset = sad__onset
        self.sad__offset = sad__offset
        self.sad__dimension = sad__dimension

        # speaker change detection hyper-parameters
        self.scd__alpha = scd__alpha
        self.scd__min_duration = scd__min_duration
        self.scd__dimension = scd__dimension

        # embedding hyper-parameters
        self.emb__internal = emb__internal

        # clustering hyper-parameters
        self.cls__damping = cls__damping
        self.cls__preference = cls__preference
        self.cls__metric = cls__metric

        step = self.feature_extraction.sliding_window().step

        # initialize speech activity detection module
        self.sad_ = Precomputed(sad__pre)
        self.sad_binarize_ = Binarize(onset=self.sad__onset,
                                      offset=self.sad__offset)

        # initialize speaker change detection module
        self.scd_ = Precomputed(scd__pre)
        self.scd_peak_ = Peak(alpha=self.scd__alpha,
                              min_duration=self.scd__min_duration,
                              percentile=False)

        # initialize speech turn embedding module
        self.emb_ = Precomputed(emb__pre)

        # initialize clustering module
        self.cls_ = my_cluster.ClusteringAP(metric=self.cls__metric,
                                        damping=self.cls__damping,
                                        preference=self.cls__preference)


    def __call__(self, current_file, annotated=False):

        # speech activity detection
        soft_sad = self.sad_(current_file)
        hard_sad = self.sad_binarize_.apply(
            soft_sad, dimension=self.sad__dimension)

        sad_output = hard_sad.to_annotation()

        # speaker change detection
        soft_scd = self.scd_(current_file)
        hard_scd = self.scd_peak_.apply(
            soft_scd, dimension=self.scd__dimension)

        # speech turns
        speech_turns = hard_scd.crop(hard_sad)

        scd_output = speech_turns.to_annotation()

        if annotated:
            speech_turns = speech_turns.crop(
                get_annotated(current_file))

        # remove small speech turns
        emb = self.emb_(current_file)
        speech_turns = [speech_turn for speech_turn in speech_turns if len(emb.crop(speech_turn, mode='loose')) > 0]

        # speech turns embedding
        to_stack = [
            np.sum(emb.crop(speech_turn, mode='loose'), axis=0)
            for speech_turn in speech_turns]
        if len(to_stack) < 1:
            return None
        fX = l2_normalize(np.vstack(to_stack))

        # speech turn clustering
        cluster_labels = self.cls_.apply(fX)

        # build hypothesis from clustering results
        hypothesis = Annotation(uri=current_file['uri'])
        for speech_turn, label in zip(speech_turns, cluster_labels):
            hypothesis[speech_turn] = label
        return hypothesis, sad_output, scd_output


class SpeakerDiarizationWeighted(object):

    def __init__(self, feature_extraction, sad__pre, scd__pre, weight__pre, emb__pre,
                 sad__onset=0.7, sad__offset=0.7, sad__dimension=1,
                 scd__alpha=0.5, scd__min_duration=1., scd__dimension=1,
                 emb__internal=False,
                 cls__damping=0.8, cls__preference=-20,
                 cls__metric='cosine'):

        super(SpeakerDiarizationWeighted, self).__init__()

        self.feature_extraction = feature_extraction

        # speech activity detection hyper-parameters
        self.sad__onset = sad__onset
        self.sad__offset = sad__offset
        self.sad__dimension = sad__dimension

        # speaker change detection hyper-parameters
        self.scd__alpha = scd__alpha
        self.scd__min_duration = scd__min_duration
        self.scd__dimension = scd__dimension

        # embedding hyper-parameters
        self.emb__internal = emb__internal

        # clustering hyper-parameters
        self.cls__damping = cls__damping
        self.cls__preference = cls__preference
        self.cls__metric = cls__metric

        step = self.feature_extraction.sliding_window().step

        # initialize speech activity detection module
        self.sad_ = Precomputed(sad__pre)
        self.sad_binarize_ = Binarize(onset=self.sad__onset,
                                      offset=self.sad__offset)

        # initialize speaker change detection module
        self.scd_ = Precomputed(scd__pre)
        self.scd_peak_ = Peak(alpha=self.scd__alpha,
                              min_duration=self.scd__min_duration,
                              percentile=False)

        # initialize weights
        self.weight_ = Precomputed(weight__pre)

        # initialize speech turn embedding module
        self.emb_ = Precomputed(emb__pre)

        # initialize clustering module
        self.cls_ = my_cluster.ClusteringAP(metric=self.cls__metric,
                                        damping=self.cls__damping,
                                        preference=self.cls__preference)


    def __call__(self, current_file, annotated=False):

        # speech activity detection
        soft_sad = self.sad_(current_file)
        hard_sad = self.sad_binarize_.apply(
            soft_sad, dimension=self.sad__dimension)

        # speaker change detection
        soft_scd = self.scd_(current_file)
        hard_scd = self.scd_peak_.apply(
            soft_scd, dimension=self.scd__dimension)

        # speech turns
        speech_turns = hard_scd.crop(hard_sad)

        if annotated:
            speech_turns = speech_turns.crop(
                get_annotated(current_file))

        # remove small speech turns
        emb = self.emb_(current_file)
        speech_turns = [speech_turn for speech_turn in speech_turns if len(emb.crop(speech_turn, mode='loose')) > 0]

        # weights
        weight = self.weight_(current_file)

        # speech turns embedding
        to_stack = [
            np.mean(emb.crop(speech_turn, mode='loose')*(1-weight.crop(speech_turn, mode='loose')), axis=0)
            for speech_turn in speech_turns]
        if len(to_stack) < 1:
            return None
        fX = l2_normalize(np.vstack(to_stack))

        # speech turn clustering
        cluster_labels = self.cls_.apply(fX)

        # build hypothesis from clustering results
        hypothesis = Annotation(uri=current_file['uri'])
        for speech_turn, label in zip(speech_turns, cluster_labels):
            hypothesis[speech_turn] = label
        return hypothesis


class SpeakerDiarizationOnSceneHAC(object):

    def __init__(self, emb__pre,
                 cls__method='average', cls__threshold=5,
                 cls__metric='cosine'):

        super(SpeakerDiarizationOnSceneHAC, self).__init__()

        # clustering hyper-parameters
        self.cls__method = cls__method
        self.cls__threshold = cls__threshold
        self.cls__metric = cls__metric

        # initialize speech turn embedding module
        self.emb_ = Precomputed(emb__pre)

        # initialize clustering module
        self.cls_ = my_cluster.ClusteringHAC(metric=self.cls__metric,
                                        method=self.cls__method,
                                        threshold=self.cls__threshold)


    def __call__(self, current_file):


        # speech turns
        hypothesis = Annotation(uri=current_file['uri'])
        sencences = current_file['speech_timeline']
        scenes = current_file['scenes']
        # remove small speech turns
        emb = self.emb_(current_file)
        #speech_turns = [speech_turn for speech_turn in speech_turns if len(emb.crop(speech_turn, mode='loose')) > 0]
        for scene in scenes:
            speech_turns = sencences.crop(scene)
            if len(speech_turns) == 0:
                continue
            if len(speech_turns) == 1:
                hypothesis[speech_turns[0]] = 1
                continue
            # speech turns embedding
            to_stack = [
                np.sum(emb.crop(speech_turn, mode='loose'), axis=0)
                for speech_turn in speech_turns]
            fX = l2_normalize(np.vstack(to_stack))

            # speech turn clustering
            cluster_labels = self.cls_.apply(fX)

            # build hypothesis from clustering results
            
            for speech_turn, label in zip(speech_turns, cluster_labels):
                hypothesis[speech_turn] = label
        return hypothesis

class SpeakerDiarizationOnEnrollHAC(object):

    def __init__(self, 
                 cls__method='average', cls__threshold=5,
                 cls__metric='cosine'):

        super(SpeakerDiarizationOnEnrollHAC, self).__init__()

        # clustering hyper-parameters
        self.cls__method = cls__method
        self.cls__threshold = cls__threshold
        self.cls__metric = cls__metric

        # initialize clustering module
        self.cls_ = my_cluster.ClusteringHAC(metric=self.cls__metric,
                                        method=self.cls__method,
                                        threshold=self.cls__threshold)


    def __call__(self, embedding, speech_turns):
        hypothesis = Annotation()
        #speech_turns = [speech_turn for speech_turn in speech_turns if len(emb.crop(speech_turn, mode='loose')) > 0]
        
        if len(speech_turns) == 0:
            return hypothesis
        if len(speech_turns) == 1:
            hypothesis[speech_turns[0]] = 1
            return hypothesis
        # speech turns embedding
        to_stack = [
            np.sum(embedding.crop(speech_turn, mode='loose'), axis=0)
            for speech_turn in speech_turns]
        fX = l2_normalize(np.vstack(to_stack))

        # speech turn clustering
        cluster_labels = self.cls_.apply(fX)

        # build hypothesis from clustering results
        
        for speech_turn, label in zip(speech_turns, cluster_labels):
            hypothesis[speech_turn] = label
        return hypothesis
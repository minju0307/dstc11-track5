from rouge_score import rouge_scorer
from summ_eval.bleu_metric import BleuMetric
from summ_eval.meteor_metric import MeteorMetric
import numpy as np


class Metric:
    def __init__(self):
        self.is_single = True
        self.reset()

    def reset(self):
        pass

    def update(self, output):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()


class DataCacheMetric(Metric):
    def __init__(self):
        self.refs = []
        self.preds = []
        super(DataCacheMetric, self).__init__()

    def reset(self):
        self.refs = []
        self.preds = []

    def update(self, output):
        hypothesis, reference = output
        assert isinstance(hypothesis, str)
        assert isinstance(reference, str)
        self.preds.append(hypothesis)
        self.refs.append(reference)

    def compute(self):
        return len(self.preds)

    def name(self):
        return "Data Count"

class BLEU(DataCacheMetric):
    def __init__(self):
        super(BLEU, self).__init__()

    def compute(self):
        if len(self.preds) == 0:
            raise ValueError("BLEU-1 must have at least one example before it can be computed!")

        metric = BleuMetric()
        score = metric.evaluate_batch(self.preds, self.refs)
        return score['bleu']

    def name(self):
        return "BLEU"


class METEOR(DataCacheMetric):
    def __init__(self):
        super(METEOR, self).__init__()

    def compute(self):
        if len(self.preds) == 0:
            raise ValueError("METEOR must have at least one example before it can be computed!")
        metric = MeteorMetric()
        score = metric.evaluate_batch(self.preds, self.refs)
        return score['meteor'] * 100


class ROUGE(Metric):
    def __init__(self):
        self.rouge_type = ['rouge1', 'rouge2', 'rougeL', "rougeLsum"]
        self.scorer = rouge_scorer.RougeScorer(self.rouge_type, use_stemmer=True)
        self._rouge = None
        self._count = None
        super(ROUGE, self).__init__()
        self.is_single = False

    def reset(self):
        self._rouge = []
        self._count = 0
        super(ROUGE, self).reset()

    def update(self, output):
        hypothesis, reference = output
        rouge = self.scorer.score(reference, hypothesis)

        _rouge = [rouge[_rouge_type].fmeasure * 100 for _rouge_type in self.rouge_type]
        self._rouge.append(_rouge)
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError("ROUGE-L must have at least one example before it can be computed!")
        return np.array(self._rouge).mean(axis=0).tolist()

    def name(self):
        return self.rouge_type

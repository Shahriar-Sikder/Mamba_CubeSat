"""Point-wise and event-wise anomaly detection metrics.

Why not point-adjust (PA) alone?  PA marks an entire ground-truth event as detected if a
single point inside it exceeds the threshold. A random scorer can then reach high F1
(Kim et al., AAAI 2022). We therefore report:

* point-wise   P / R / F1 (at threshold), AUROC, AUPRC (threshold-free)
* event-wise   - event recall    = fraction of GT events with >= 1 alarm inside
               - event precision = TP events / (TP events + false-alarm episodes), where a
                 false-alarm episode is a maximal run of alarms on nominal points
               - event F1, event AUPRC (threshold sweep of the event P/R curve)
               - event AUROC: area under (nominal-point FPR, event recall) as the
                 threshold is swept, i.e. "fraction of incidents caught vs false-alarm rate"
               Note: any ">= 1 alarm inside" definition favours long events, so a random
               scorer gets event AUROC > 0.5 when events are long. Compare to a random baseline.
* composite F1 (Garg et al., 2021): event recall combined with point-wise precision
* point-adjusted F1 is returned ONLY as a reference, to show how optimistic it is.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_from_nominal(scores, labels, q=99.0):
    """Threshold = q-th percentile of scores on NOMINAL points (e.g. validation split)."""
    nominal = np.asarray(labels) == 0
    return float(np.percentile(scores[nominal], q))


def runs(mask):
    """Maximal runs of True -> list of (start, end) with end exclusive."""
    m = np.asarray(mask).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def events_from_ids(event_id):
    """GT events as (start, end) using event ids (keeps adjacent events separate)."""
    event_id = np.asarray(event_id)
    out = []
    for eid in np.unique(event_id[event_id >= 0]):
        idx = np.where(event_id == eid)[0]
        out.append((int(idx[0]), int(idx[-1]) + 1, int(eid)))
    return out


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


# --------------------------------------------------------------------------------------
def pointwise_metrics(scores, labels, thr):
    labels = np.asarray(labels).astype(int)
    pred = scores > thr
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    p, r, f = _prf(tp, fp, fn)
    return dict(precision=p, recall=r, f1=f,
                auroc=roc_auc_score(labels, scores), auprc=average_precision_score(labels, scores),
                tp=tp, fp=fp, fn=fn)


def _event_counts(pred, labels, gt_events):
    detected = np.array([pred[s:e].any() for s, e, _ in gt_events], dtype=bool)
    fa_episodes = len(runs(pred & (labels == 0)))
    return detected, fa_episodes


def _sweep_thresholds(scores):
    """Quantile thresholds, dense near the top where false-alarm rates are small."""
    q = np.r_[np.linspace(0.0, 0.9, 91), 1 - np.logspace(-1, -4.5, 300), 1.0]
    return np.unique(np.quantile(scores, q))[::-1]          # high -> low


def eventwise_metrics(scores, labels, event_id, thr):
    labels = np.asarray(labels).astype(int)
    nominal = labels == 0
    gt = events_from_ids(event_id)
    pred = scores > thr
    detected, fa = _event_counts(pred, labels, gt)
    tp = int(detected.sum())
    p, r, f = _prf(tp, fa, len(gt) - tp)

    # composite F1: event recall + point precision
    pt_tp = int((pred & (labels == 1)).sum())
    pt_p = pt_tp / max(int(pred.sum()), 1)
    comp_f1 = 2 * pt_p * r / (pt_p + r) if pt_p + r > 0 else 0.0

    # threshold sweep (high -> low): event P/R curve and (nominal FPR, event recall) curve
    thrs = _sweep_thresholds(scores)
    P, R, FPR = [], [], []
    for t in thrs:
        pr_t = scores > t
        d, fa_t = _event_counts(pr_t, labels, gt)
        tp_t = int(d.sum())
        pp, rr, _ = _prf(tp_t, fa_t, len(gt) - tp_t)
        P.append(1.0 if tp_t + fa_t == 0 else pp)
        R.append(rr)
        FPR.append((pr_t & nominal).sum() / max(nominal.sum(), 1))
    P, R, FPR = np.array(P), np.array(R), np.array(FPR)
    auprc = float(np.sum(np.diff(np.concatenate([[0.0], R])) * P))
    fx, ry = np.r_[0.0, FPR, 1.0], np.r_[0.0, R, 1.0]
    auroc = float(np.sum(np.diff(fx) * (ry[1:] + ry[:-1]) / 2))

    per_event = [dict(id=eid, start=s, end=e, detected=bool(dd), max_score=float(scores[s:e].max()))
                 for (s, e, eid), dd in zip(gt, detected)]
    return dict(precision=p, recall=r, f1=f, auroc=auroc, auprc=auprc, composite_f1=comp_f1,
                n_events=len(gt), detected=tp, false_alarm_episodes=fa,
                curve=dict(precision=P, recall=R, fpr=FPR, thresholds=thrs), per_event=per_event)


def point_adjusted_f1(scores, labels, event_id, thr):
    """REFERENCE ONLY - known to be over-optimistic."""
    labels = np.asarray(labels).astype(int)
    pred = scores > thr
    adj = pred.copy()
    for s, e, _ in events_from_ids(event_id):
        if pred[s:e].any():
            adj[s:e] = True
    tp = int((adj & (labels == 1)).sum())
    fp = int((adj & (labels == 0)).sum())
    fn = int((~adj & (labels == 1)).sum())
    return _prf(tp, fp, fn)[2]


def full_report(scores, labels, event_id, thr, **kw):
    pw = pointwise_metrics(scores, labels, thr)
    ew = eventwise_metrics(scores, labels, event_id, thr, **kw)
    pa = point_adjusted_f1(scores, labels, event_id, thr)
    return dict(point=pw, event=ew, pa_f1=pa, threshold=thr)


def format_report(rep, name=""):
    pw, ew = rep["point"], rep["event"]
    lines = [f"=== {name}  (threshold = {rep['threshold']:.4f}) ===",
             f"{'':12s}{'P':>8s}{'R':>8s}{'F1':>8s}{'AUROC':>8s}{'AUPRC':>8s}",
             f"{'point-wise':12s}{pw['precision']:8.3f}{pw['recall']:8.3f}{pw['f1']:8.3f}"
             f"{pw['auroc']:8.3f}{pw['auprc']:8.3f}",
             f"{'event-wise':12s}{ew['precision']:8.3f}{ew['recall']:8.3f}{ew['f1']:8.3f}"
             f"{ew['auroc']:8.3f}{ew['auprc']:8.3f}",
             f"events detected {ew['detected']}/{ew['n_events']}, "
             f"false-alarm episodes {ew['false_alarm_episodes']}, "
             f"composite F1 {ew['composite_f1']:.3f}",
             f"(reference only) point-adjusted F1 {rep['pa_f1']:.3f}"]
    return "\n".join(lines)


def per_type_recall(per_event, events):
    """Event recall grouped by anomaly kind (point / contextual / collective)."""
    kinds = {e["id"]: e["kind"] for e in events}
    out = {}
    for pe in per_event:
        k = kinds[pe["id"]]
        d, n = out.get(k, (0, 0))
        out[k] = (d + int(pe["detected"]), n + 1)
    return {k: f"{d}/{n}" for k, (d, n) in out.items()}

"""Independent retained Decimal estimator for native migration checks."""
from decimal import Decimal, localcontext

def estimate(results, score=lambda result: (1, 0)):
    """Weighted acceptance and joint diagnostic means; zero weights count.

    score returns acceptance in [0,1], conditional diagnostic in [0,1].
    Ratios are reported with delta-method standard errors, not as unbiased.
    A failed proposal invalidates the aggregate instead of removing a sample.
    """
    with localcontext() as ctx:
        ctx.prec = 80
        n = 0; zeros = 0; failures = []; sums = [Decimal(0)]*5; sw = sw2 = Decimal(0)
        for result in results:
            n += 1
            if result.compiled is None:
                failures.append(dict(sample_id=result.assignment.sample_id, status=result.status, error=result.error)); continue
            w = result.weight_decimal(); zeros += int(not w); sw += w; sw2 += w*w
            accept, diagnostic = score(result)
            a = Decimal(str(accept)); d = Decimal(str(diagnostic))
            if not (0 <= a <= 1 and 0 <= d <= 1): raise ValueError('scores must lie in [0,1]')
            x = w*a; y = x*d
            for j, v in enumerate((x, y, x*x, y*y, x*y)): sums[j] += v
        if not n: raise ValueError('at least one proposal required')
        base = dict(proposals=n, zero_weights=zeros, failures=failures,
                    status='incomplete' if failures else 'completed', ratio_unbiased=False)
        if failures: return base
        sx, sy, xx, yy, xy = sums; mean = sx/n; joint = sy/n
        def se(s, s2):
            return float((max(Decimal(0), (s2-s*s/n)/(n*(n-1)))).sqrt()) if n > 1 else None
        ratio = sy/sx if sx else None
        ratio_se = (float((max(Decimal(0), yy-2*ratio*xy+ratio*ratio*xx)/(n*(n-1))).sqrt()/mean)
                    if ratio is not None and n > 1 else None)
        return dict(base, acceptance=float(mean), acceptance_standard_error=se(sx, xx),
                    joint_diagnostic=float(joint), joint_standard_error=se(sy, yy),
                    conditional_diagnostic=float(ratio) if ratio is not None else None,
                    conditional_standard_error=ratio_se,
                    mean_weight=float(sw/n), effective_sample_size=float(sw*sw/sw2) if sw2 else 0.,
                    uncertainty_status='empirical standard errors only' if sw else
                    'no nonzero weights observed; zero empirical variance does not establish zero probability')

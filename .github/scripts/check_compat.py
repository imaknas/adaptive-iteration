"""The adaptive-iteration shim must keep every old import path working."""
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import adaptive_iteration
    import adaptive_iteration.core.ledger as old_ledger
    from adaptive_iteration import Evaluator, Ledger, PairedProportionRule  # noqa: F401
    from adaptive_iteration.core.decision import WelchIntervalRule  # noqa: F401
    from adaptive_iteration.core.domain import DomainConfig  # noqa: F401
    from adaptive_iteration.core.shrinkage import approx_se, estimate_prior  # noqa: F401
    from adaptive_iteration.migrate import v1_to_v2  # noqa: F401
    from adaptive_iteration.replay import calibrate, replay  # noqa: F401

import ordal

assert any(issubclass(w.category, DeprecationWarning) for w in caught), "no deprecation warning"
assert old_ledger.Ledger is ordal.Ledger
assert adaptive_iteration.__version__ == ordal.__version__
print("compat shim ok:", adaptive_iteration.__version__)

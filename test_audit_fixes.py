"""验证 B1/B2/B3/B5 四个 bug 修复 / Verify audit bug fixes."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import inspect
from datetime import datetime, timedelta

from src.tsf_frame.models.transformer.transformer_models import (
    iTransformer, TimesNet, _dl_fit,
)
from src.tsf_frame.monitoring.stores import InMemoryStore


def test_b1_itransformer_seqlen():
    """B1: iTransformer should raise if input seq_len != config seq_len."""
    print("=== B1: iTransformer seq_len validation ===")
    m = iTransformer({'input_size': 3, 'seq_len': 24, 'output_size': 1})
    x_ok = torch.randn(2, 24, 3)
    out = m(x_ok)
    print(f"  OK shape (24 match 24): {out.shape}")
    assert out.shape == (2, 1)

    try:
        m(torch.randn(2, 10, 3))  # L=10 != 24
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  CAUGHT (10 vs 24): {str(e)[:80]}...")
    print("  [OK] B1 PASS\n")


def test_b2_timesnet_min_seqlen():
    """B2: TimesNet should reject seq_len < min threshold."""
    print("=== B2: TimesNet min seq_len guard ===")
    try:
        TimesNet({'seq_len': 4, 'input_size': 1})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  CAUGHT (seq_len=4): {str(e)[:80]}...")

    m = TimesNet({'seq_len': 24, 'input_size': 1})
    print(f"  OK (seq_len=24): model={m.model_name}")
    print("  [OK] B2 PASS\n")


def test_b3_val_preupload():
    """B3: _dl_fit should pre-upload val data outside epoch loop."""
    print("=== B3: Validation data pre-upload ===")
    src = inspect.getsource(_dl_fit)
    assert 'X_val_t' in src, "Pre-upload variable X_val_t not found"
    assert 'y_val_t' in src, "Pre-upload variable y_val_t not found"
    # Ensure the old pattern is gone
    assert 'X_val = torch.FloatTensor(X_val).to(device)' not in src, \
        "Old in-loop upload pattern still present"
    print("  Pre-upload vars present: X_val_t, y_val_t")
    print("  Old in-loop pattern removed")
    print("  [OK] B3 PASS\n")


def test_b5_inmemory_index_rebuild():
    """B5: InMemoryStore should rebuild _pred_index after cleanup_old."""
    print("=== B5: InMemoryStore index rebuild after cleanup ===")
    s = InMemoryStore()
    old_ts = datetime.now() - timedelta(days=100)
    new_ts = datetime.now()

    s.insert_prediction(model_id='m', timestamp=old_ts, target='y', y_pred=1.0)
    s.insert_prediction(model_id='m', timestamp=new_ts, target='y', y_pred=2.0)
    print(f"  Before: {len(s._predictions)} preds, {len(s._pred_index)} index keys")
    assert len(s._predictions) == 2

    removed = s.cleanup_old(retain_days=30)
    print(f"  After cleanup: {len(s._predictions)} preds, removed={removed}")
    assert len(s._predictions) == 1
    assert removed >= 1

    # The critical test: update_actual should hit the CORRECT row
    s.update_actual(model_id='m', timestamp=new_ts, target='y', y_actual=2.5)
    actual = s._predictions[0].get('y_actual')
    print(f"  After update_actual, y_actual={actual}")
    assert actual == 2.5, f"Expected 2.5, got {actual}"

    # Verify index is consistent
    key = ('m', 'y', new_ts)
    assert key in s._pred_index, "Index should contain the surviving record"
    assert s._pred_index[key] == [0], f"Index should point to position 0, got {s._pred_index[key]}"
    print("  [OK] B5 PASS\n")


if __name__ == '__main__':
    test_b1_itransformer_seqlen()
    test_b2_timesnet_min_seqlen()
    test_b3_val_preupload()
    test_b5_inmemory_index_rebuild()
    print("=" * 50)
    print("ALL VERIFICATIONS PASSED [OK]")

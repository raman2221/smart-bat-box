"""
combine_models.py

Blends two fine-tuned detectors into one ("model soup"): every weight of
the new model is a mix of the same weight in model A and model B:

    blended = (1 - mix) * A  +  mix * B

This works because v4 was trained starting from v3, so the two models are
close relatives; blending them often keeps the strengths of both (v3 finds
more bats, v4 is better at ignoring people, birds and lamps). The result
is a single model, so it runs exactly as fast as either one.

Run from the project folder (with the .venv active):
    python training/combine_models.py models/detection/finetuned_detector_v3.pt models/detection/finetuned_detector_v4.pt --mix 0.3 0.5 0.7

This saves e.g. models/detection/finetuned_detector_blend50.pt (mix 0.5).
Compare them with training/evaluate.py.
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def blend(state_a, state_b, mix):
    """Mix two models' weights. Whole-number counters are copied from A."""
    blended = OrderedDict()
    for name, a in state_a.items():
        b = state_b[name]
        if a.is_floating_point():
            blended[name] = (1 - mix) * a + mix * b
        else:
            blended[name] = a.clone()   # e.g. counters: can't be averaged
    # A saved model also carries hidden per-layer version notes. Without
    # them PyTorch treats the file as an old format and renames layers when
    # loading, so keep A's notes.
    if hasattr(state_a, "_metadata"):
        blended._metadata = state_a._metadata
    return blended


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_a", type=Path, help="first weights file (mix 0 = all A)")
    parser.add_argument("model_b", type=Path, help="second weights file (mix 1 = all B)")
    parser.add_argument("--mix", type=float, nargs="+", default=[0.5],
                        help="share of model B, between 0 and 1 (several allowed)")
    args = parser.parse_args()

    state_a = torch.load(args.model_a, map_location="cpu")
    state_b = torch.load(args.model_b, map_location="cpu")
    if state_a.keys() != state_b.keys():
        raise SystemExit("The two models have different layers and can't be blended.")

    for mix in args.mix:
        if not 0 <= mix <= 1:
            raise SystemExit(f"--mix must be between 0 and 1, got {mix}")
        out = args.model_a.parent / f"finetuned_detector_blend{round(mix * 100):02d}.pt"
        torch.save(blend(state_a, state_b, mix), out)
        print(f"mix {mix:.2f} -> {out}")


if __name__ == "__main__":
    main()

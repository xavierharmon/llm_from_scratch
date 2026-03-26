"""
Beam Search Decoder
====================
Maintains the top-B most likely sequences at each step rather than
greedily committing to the single best token.

When to use:
    - When you want more coherent, higher-probability outputs
    - Translation, summarization, structured generation
    - Slower than greedy/sampling but higher average quality

Algorithm:
    1. Start with B identical copies of the prompt (beams)
    2. For each beam, compute next-token logits
    3. Expand each beam by all vocab tokens → B × vocab_size candidates
    4. Keep only the top-B candidates by cumulative log-probability
    5. Repeat until all beams hit EOS or max length
    6. Return the highest-scoring complete beam

Tradeoff vs sampling:
    Beam search finds higher-probability text, but can be repetitive
    and "safe." Sampling (nucleus) is more diverse and surprising.
    For creative running log generation, sampling is usually better.
    For evaluation/benchmarking, beam search gives reproducible outputs.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Beam:
    """A single hypothesis in the beam."""
    token_ids: list[int]
    log_prob: float = 0.0
    finished: bool = False


@torch.no_grad()
def beam_search(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    beam_size: int = 4,
    max_new_tokens: int = 80,
    eos_id: Optional[int] = None,
    length_penalty: float = 1.0,
    device: Optional[torch.device] = None,
) -> list[int]:
    """
    Beam search decoding.

    Args:
        model:          trained model
        token_ids:      [1, T] prompt token IDs
        beam_size:      number of beams to maintain
        max_new_tokens: maximum tokens to generate
        eos_id:         end-of-sequence token ID
        length_penalty: > 1.0 favors longer sequences, < 1.0 shorter
        device:         torch device

    Returns:
        Best sequence of token IDs (list of ints, prompt + generated)
    """
    model.eval()
    if device is not None:
        token_ids = token_ids.to(device)

    prompt_list = token_ids[0].tolist()
    beams = [Beam(token_ids=prompt_list.copy(), log_prob=0.0)]
    max_seq = model.max_seq_len

    for step in range(max_new_tokens):
        if all(b.finished for b in beams):
            break

        candidates = []

        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue

            ctx = torch.tensor(beam.token_ids[-max_seq:]).unsqueeze(0)
            if device is not None:
                ctx = ctx.to(device)

            logits, _ = model(ctx)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

            top_log_probs, top_ids = torch.topk(log_probs, beam_size)

            for log_p, tok_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                new_ids = beam.token_ids + [tok_id]
                finished = (eos_id is not None and tok_id == eos_id)
                # Length-penalized score
                length = len(new_ids) - len(prompt_list)
                score = (beam.log_prob + log_p) / max(1, length) ** length_penalty
                candidates.append(Beam(
                    token_ids=new_ids,
                    log_prob=beam.log_prob + log_p,
                    finished=finished,
                ))

        # Keep top beam_size candidates by score
        candidates.sort(
            key=lambda b: b.log_prob / max(1, len(b.token_ids) - len(prompt_list)) ** length_penalty,
            reverse=True,
        )
        beams = candidates[:beam_size]

    # Return the best finished beam (or best overall if none finished)
    finished = [b for b in beams if b.finished]
    best = finished[0] if finished else beams[0]
    return best.token_ids

from functools import cache

import torch
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadata

from brrr import cuext

MAX_ROWS = 128

type Indexed = tuple[torch.Tensor, int, torch.Tensor, DeepseekV41B12xMetadata]


@cache
def partial(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ext = cuext.extension("sparse_decode")
    return (
        torch.empty(MAX_ROWS * ext.CL * ext.H * ext.D, dtype=torch.float32, device=device),
        torch.empty(MAX_ROWS * ext.CL * ext.H * 2, dtype=torch.float32, device=device),
    )


def fits(q: torch.Tensor, swa_width: int, indexed_width: int) -> bool:
    ext = cuext.extension("sparse_decode")
    total = swa_width + indexed_width
    return (
        q.shape[0] <= MAX_ROWS
        and q.shape[1:] == (ext.H, ext.D)
        and q.dtype == torch.bfloat16
        and total % (ext.CL * 16) == 0
        and total // ext.CL <= ext.MAXK
    )


def decode(
    q: torch.Tensor,
    out: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_page: int,
    swa: DeepseekV41B12xMetadata,
    window: int,
    swa_width: int,
    draft: bool,
    replay: torch.Tensor | None,
    indexed: Indexed | None,
    sink: torch.Tensor,
    sm_scale: float,
) -> None:
    match indexed:
        case None:
            index = (None, 0, None, None, None, None)
        case (idx_cache, idx_page, idx_logical, main):
            index = (idx_cache, idx_page, idx_logical, main.cache_lengths, main.req_id_per_token, main.block_table)
    cuext.extension("sparse_decode").decode(
        q, out, swa_cache, swa_page, swa.positions, swa.req_id_per_token, swa.block_table, swa.query_start_loc, swa.request_positions, replay,
        window, swa_width, draft, *index, sink, sm_scale, *partial(q.device),
    )

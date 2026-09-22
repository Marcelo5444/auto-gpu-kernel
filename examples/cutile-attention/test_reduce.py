import cuda.tile as ct
import torch

@ct.kernel
def test_reduce(x: ct.Array, out: ct.Array):
    bid = ct.bid(0)
    # x is [B, M, N] = [2, 4, 4]
    tile = ct.load(x, index=(bid, 0, 0), shape=(1, 4, 4)).reshape((4, 4))
    # Try reduce with literal axis
    m = ct.max(tile, axis=1, keepdims=True)
    ct.store(out, index=(bid, 0, 0), tile=m.reshape((1, 4, 1)))

x = torch.randn(2, 4, 4).cuda().half()
out = torch.empty(2, 4, 1).cuda().half()
ct.launch(torch.cuda.current_stream().cuda_stream, (2,), test_reduce, (x, out))
print('out:', out)
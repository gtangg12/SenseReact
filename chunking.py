import numpy as np

CHUNK_PENALTY = 10

def chunk(frame_embeds):
    """
    Given a tensor of frame embeddings (of size n_frames * embed_dim), chunk them into contiguous sections
    in which the embeddings are similar.

    Use dynamic programming to find the optimal chunking. The cost of a chunking is
    the sum of variance of embeddings within each chunk, plus a penalty for the number of chunks.
    """
    n_frames = frame_embeds.shape[0]

    costs = np.zeros(n_frames+1)
    parents = np.zeros(n_frames+1, dtype=np.int)
    for i in range(1, n_frames +1):
        possible_costs = [
            CHUNK_PENALTY + costs[j] + (i-j)*torch.var(frame_embeds[j:i], axis=0).mean() for j in range(max(i-60, 0), i-1)
        ]
        possible_costs.extend([CHUNK_PENALTY + costs[i-1]]) # since torch.var doesn't work for singleton chunks
        
        parents[i] = np.argmin(possible_costs)
        costs[i] = possible_costs[parents[i]]

    # Reconstruct the optimal chunking
    chunks = []
    i = n_frames
    while i > 0:
        chunks.append((parents[i], i))
        i = parents[i]
    
    chunks.reverse()

    return chunks


if __name__ == "__main__":
    import torch

    frame_embeds = torch.tensor([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [10, 15, 14],
        [10, 15, 14],
        [10, 15, 14],
    ], dtype=torch.float)

    frame_embeds = torch.tile(frame_embeds, (4, 1))
    print(frame_embeds)

    frame_embeds += torch.rand_like(frame_embeds) * 5
    print(torch.var(frame_embeds, axis=0).mean())

    chunks = chunk(frame_embeds)

    print(chunks)






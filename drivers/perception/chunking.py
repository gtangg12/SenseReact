import numpy as np

CHUNK_PENALTY = 0.3

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
            CHUNK_PENALTY + costs[j] + (i-j)*torch.var(frame_embeds[j:i], axis=0).mean() for j in range(max(i-90, 0), i-1)
        ]
        possible_costs.extend([CHUNK_PENALTY + costs[i-1]]) # since torch.var doesn't work for singleton chunks
        
        parents[i] = np.argmin(possible_costs) + max(i-90, 0)
        costs[i] = possible_costs[parents[i]-max(i-90,0)]
        print('chunked', i)        

    # Reconstruct the optimal chunking
    chunks = []
    i = n_frames
    while i > 0:
        chunks.append((parents[i], i))
        i = parents[i]
    
    chunk_embeds = []
    chunks.reverse()
    for (start, end) in chunks:
        chunk_embeds.append(frame_embeds[start:end].mean(axis=0))
    
    chunk_embeds = torch.stack(chunk_embeds, axis=0)

    return chunks, chunk_embeds


if __name__ == "__main__":
    import torch

    frame_embeds = torch.tensor(np.load('embeddings_sample.npy'))

    chunks, chunk_embeds = chunk(frame_embeds)

    print(chunks)

    np.save('chunk_embeds_sample.npy', chunk_embeds.numpy())
    np.save('chunks_sample.npy', np.array(chunk))








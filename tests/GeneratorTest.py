from salsaclrs import SALSACLRSDataset
ds = SALSACLRSDataset(root="./", split="train", algorithm="bfs", num_samples=10, graph_generator="er", graph_generator_kwargs={"n": [16, 32], "p_range": (0.1, 0.3)}, hints=True)
from salsaclrs import SALSACLRSDataLoader
dl = SALSACLRSDataLoader(ds, batch_size=32, num_workers=6)

print(dl.dataset)
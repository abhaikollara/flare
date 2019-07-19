def split(datasets, split_ratio=0.2):
    lens = {len(x) for x in datasets}
    if len(lens) != 1:
        raise ValueError(f"All datasets must be of the same size. Found sizes {lens}")
    split_point = int(len(datasets[0]) * 0.2)

    return [(x[split_point:],x[:split_point]) for x in datasets]
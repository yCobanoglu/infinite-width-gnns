from gnn.transforms.sparsify_transforms import RandomSparsifyGraph, EffectiveResistance


def select_sparsification(sparsify):
    match sparsify.name:
        case "random":
            return RandomSparsifyGraph(sparsify.rate)
        case "ef":
            return EffectiveResistance(sparsify.rate)
        case _:
            raise ValueError(f"Sparsification {sparsify} not implemented")

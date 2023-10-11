from collections import OrderedDict

import git
from tabulate import tabulate


def unique_key(key, value, existing):
    if key in existing.keys() and str(value) == existing[key]:
        counter = len([name for name in existing.keys() if key in name])
        return f"{key}-{counter}"
    return key


def unique_key_model(key, value, existing):
    if key in existing.keys() and str(value) != existing[key]:
        counter = len([key in name for name in existing.keys()])
        return f"{key}-{counter}"
    return key


def summarize(datasets, configs, PATH, num_runs):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    datasets_by_unique_name = OrderedDict()
    for dataset in datasets:
        v = str(datasets)
        unique = unique_key(dataset.summarize_name, v, datasets_by_unique_name)
        datasets_by_unique_name[unique] = v
    columns = list(datasets_by_unique_name.keys())

    rows_concise = {}
    rows_full = {}
    models_unique = {}

    for counter, config in enumerate(configs):
        if counter % len(datasets) == 0 or counter == 0:
            model = config.model
            model.parameters = None
            v = str(model)
            unique = unique_key_model(model.summarize_name, v, models_unique)
            models_unique[unique] = v
        rows_concise.setdefault(unique, []).append(config.metrics.to_table(is_concise=True))
        rows_full.setdefault(unique, []).append(config.metrics.to_table(is_concise=False))

    pd_concise = tabulate([[k, *v] for k, v in rows_concise.items()], headers=columns, tablefmt="grid")
    pd_full = tabulate([[k, *v] for k, v in rows_full.items()], headers=columns, tablefmt="grid")

    with open(PATH.parent / "results.txt", "w") as f:
        f.write(f"Commit: {sha}\n\n")
        f.write(f"Number of runs per experiment: {num_runs}\n\n")
        f.write("Very concise format\n\n")
        f.write(pd_concise)
        f.write("\n\nSlightly more verbose\n")
        f.write(pd_full)

    print(pd_full)

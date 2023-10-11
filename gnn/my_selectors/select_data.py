import warnings
from typing import Sequence

from torch_geometric.datasets import (
    Planetoid,
    NELL,
    Amazon,
    Coauthor,
    TUDataset,
    FAUST,
    CoMA,
    GitHub,
    Reddit,
    TOSCA,
    ShapeNet,
    ModelNet,
    LRGBDataset,
    WebKB,
    KarateClub,
    AttributedGraphDataset,
    WikipediaNetwork,
)
from torch_geometric.transforms import (
    FaceToEdge,
    ToUndirected,
    Compose,
    ToDevice,
    Constant,
)

from gnn.datasets.dataset_split import split_data_inductive
from gnn.datasets.generator_dataset import GeneratorDataset
from gnn.datasets.synthetic_data import (
    sbm,
    sbm_classification_generator,
    barasi_albert,
    barasi_albert_classification,
)
from gnn.device import DEVICE
from gnn.transforms.basic_transforms import (
    FeaturesToSparse,
    ConstantFeaturesAllOnes,
    IdentitiyFeatures,
    RemoveSelfLoops,
    SetAdj,
    TransductiveDataSplit,
    ValidateInductive,
    ToFloat,
    FeaturesToDense,
    OneHotToVector,
)


def select_data(data_config, epochs, path_root="/tmp", transforms: Sequence[any] = tuple()):
    print("Dataset: ", data_config.name)
    root = f"{path_root}/{data_config.name}"
    # root = f"pytorchgeometric_data/{data_config.name}"
    name = data_config.name
    t = [
        ToFloat(),
        ToUndirected(),
        RemoveSelfLoops(),
        SetAdj(),
        *transforms,
        ToDevice(DEVICE),
    ]

    match name:
        case "reddit-binary" | "imdb-binary" | "collab":
            if not isinstance(data_config.feature_dim, int):
                warnings.warn(
                    "Need Feature Dimension because there are not features on this dataset and IdentitiyFeatures won't work because feature dim has to the same (and for graphes of different sizes it is not)"
                )
                t_ = Compose([Constant(), *t])
            else:
                t_ = Compose([*t])
            dataset = TUDataset(root=root, name=name.upper(), transform=t_)
        case "proteins" | "enzymes" | "mutag":
            dataset = TUDataset(root=root, name=name.upper(), transform=Compose(t))
        case "wiki" | "blogcatalog" | "facebook" | "flickr":
            names = {
                "wiki": "Wiki",
                "blogcatalog": "BlogCatalog",
                "facebook": "Facebook",
                "flickr": "Flickr",
            }
            t = [*t[:-1]]
            if name != "wiki":
                t.append(OneHotToVector)
            t.append(ToDevice(DEVICE))
            dataset = AttributedGraphDataset(root=root, name=names[name], transform=Compose(t))
        case "amazon_photo" | "amazon_computers":
            dataset = Amazon(root=root, name=name.split("_")[1], transform=Compose(t))
        case "coma":
            t_ = Compose([IdentitiyFeatures(), *t])
            dataset = CoMA(root="tmp/coma", transform=t_, pre_transform=FaceToEdge())
        case "Peptides-func":
            train_dataset = LRGBDataset(root=root, name="Peptides-func", split="train", transform=Compose(t))
            val_dataset = LRGBDataset(root=root, name="Peptides-func", split="val", transform=Compose(t))
            test_dataset = LRGBDataset(root=root, name="Peptides-func", split="test", transform=Compose(t))
            return (train_dataset, val_dataset, test_dataset)
        case "tosca":
            dataset = TOSCA(root=root, transform=Compose(t), pre_transform=FaceToEdge())
        case "shapenet":
            t_ = Compose([ConstantFeaturesAllOnes(num_features=data_config.feature_dim), *t])
            train_dataset = ShapeNet(
                root=root,
                categories=None,
                pre_transform=FaceToEdge(),
                include_normals=True,
                split="train",
                transform=t_,
            )
            val_dataset = ShapeNet(
                root=root,
                categories=None,
                pre_transform=FaceToEdge(),
                include_normals=True,
                split="val",
                transform=t_,
            )
            test_dataset = ShapeNet(
                root=root,
                categories=None,
                pre_transform=FaceToEdge(),
                include_normals=True,
                split="test",
                transform=t_,
            )
            return (train_dataset, val_dataset, test_dataset)
        case "modelnet":
            t_ = Compose([ConstantFeaturesAllOnes(num_features=data_config.feature_dim), *t])
            train_dataset = ModelNet(
                root=root,
                pre_transform=FaceToEdge(),
                name="10",
                train=True,
                transform=t_,
            )
            test_dataset = ModelNet(
                root=root,
                pre_transform=FaceToEdge(),
                name="10",
                train=False,
                transform=t_,
            )
            dataset = (train_dataset, None, test_dataset)
        case "faust":
            t_ = Compose([IdentitiyFeatures(), *t])
            train_dataset = FAUST(root="tmp/faust", pre_transform=FaceToEdge(), transform=t_)
            test_dataset = FAUST(root="tmp/faust", train=False, pre_transform=FaceToEdge(), transform=t_)
            return (train_dataset, None, test_dataset)
        case "nell":
            # normalization needs dense features, nell features are sparse
            # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.NELL.html#torch_geometric.datasets.NELL
            t_ = Compose([FeaturesToDense(), *t, FeaturesToSparse()])
            dataset = NELL(root=root, transform=t_)
        case "coauthor_cs" | "coauthor_phy":
            t_ = Compose([FeaturesToSparse(), *t])
            name = {"coauthor_cs": "CS", "coauthor_phy": "Physics"}[name]
            dataset = Coauthor(root=root, name=name, transform=t_)
        case "cornell" | "texas" | "wisconsin":
            dataset = WebKB(root=root, name=name.upper(), transform=Compose(t))
        case "karate":
            dataset = KarateClub(transform=Compose(t))
        case "reddit":
            dataset = Reddit(root=root, transform=Compose(t))
        case "github":
            dataset = GitHub(root=root, transform=Compose(t))
        case "chameleon_c" | "squirrel_c":
            name = {"chameleon_c": "Chameleon", "squirrel_c": "Squirrel"}[name]
            dataset = WikipediaNetwork(root=root, name=name, transform=Compose(t), geom_gcn_preprocess=True)
        case "chameleon" | "squirrel" | "crocodile":
            # dataset = MyWikipediaNetwork(root=root, name=name, transform=Compose(t))
            dataset = WikipediaNetwork(root=root, name=name, transform=Compose(t), geom_gcn_preprocess=False)
        case "cora" | "citeseer" | "pubmed":
            name = {"pubmed": "PubMed", "cora": "Cora", "citeseer": "CiteSeer"}[name]
            dataset = Planetoid(root=root, name=name, transform=Compose(t))
        case "sbm_classification":
            n_classes = len(data_config.graphs)
            generator = lambda: sbm_classification_generator(
                data_config.graphs,
                data_config.nodes,
                Compose(t),
                data_config.feature_dim,
            )
            dataset = GeneratorDataset(generator, data_config.samples, to_store=True, num_classes=n_classes)
        case "sbm":
            generator = lambda: sbm(data_config.nodes, data_config.prob_matrix, data_config.feature_dim, t)
            STORE_GENERTOR = epochs != 1
            dataset = GeneratorDataset(generator, data_config.samples, to_store=STORE_GENERTOR)
        case "barasi-albert":
            t_ = Compose([IdentitiyFeatures(), *t])
            dataset = [barasi_albert(data_config.nodes, data_config.edges, t_)]
        case "barasi-albert_classification":
            t_ = Compose([IdentitiyFeatures(), *t])
            dataset = barasi_albert_classification(data_config.graphs, data_config.samples, t_)
        case _:
            raise ValueError(f"Unknown dataset {name}")
    return dataset


def select_dataset(
    config_data,
    learning_task,
    epochs,
    transforms=tuple(),
):
    match learning_task:
        case "transductive":
            dataset = select_data(
                config_data,
                epochs=epochs,
                transforms=[
                    TransductiveDataSplit(config_data.data_split, config_data.unlabelled),
                    *transforms,
                ],
            )
            if len(dataset) != 1:
                raise ValueError("Transductive should return a single dataset")
            return [
                dataset,
                dataset if hasattr(dataset[0], "val_mask") else None,
                dataset if hasattr(dataset[0], "test_mask") else None,
            ]  # training method expects train, val, test datasets even for transductive setting
        case "link prediction":
            return select_data(config_data, epochs=epochs, transforms=transforms)
        case "inductive" | "inductive_clustering":
            dataset = select_data(
                config_data,
                epochs=epochs,
                transforms=[ValidateInductive(), *transforms],
            )
            if isinstance(dataset, tuple):
                return dataset
            dataset = split_data_inductive(config_data.data_split, dataset)
        case _:
            raise ValueError(f"Learning task '{learning_task}' not found")
    return dataset

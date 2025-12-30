"""
Network Dataset Loaders for INTERCEPT Experiments.

This module provides utilities for creating synthetic networks and loading
real-world datasets for cascade control experiments.

Synthetic Networks:
    - Barabási-Albert (scale-free)
    - Erdős-Rényi (random)
    - Watts-Strogatz (small-world)
    - Powerlaw cluster graphs

Real-World Datasets:
    - SNAP datasets (Email-Eu-core, ca-GrQc, etc.)
    - NetworkX built-in social networks

Usage:
    >>> from src.network_datasets import get_all_test_networks
    >>> 
    >>> networks = get_all_test_networks(include_snap=True)
    >>> for name, graph in networks.items():
    ...     print(f"{name}: {graph.number_of_nodes()} nodes")

References:
    - Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks.
    - Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks.
    - Leskovec, J., & Krevl, A. (2014). SNAP Datasets: Stanford Large Network Dataset Collection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np


def create_barabasi_albert(
    n: int = 200,
    m: int = 3,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Create Barabási-Albert preferential attachment graph.
    
    Scale-free network where new nodes attach preferentially to
    high-degree nodes. Models social networks, citation networks,
    and the web.
    
    Args:
        n: Number of nodes
        m: Number of edges to attach from each new node
        seed: Random seed for reproducibility
    
    Returns:
        NetworkX Graph
    """
    return nx.barabasi_albert_graph(n, m, seed=seed)


def create_erdos_renyi(
    n: int = 200,
    p: float = 0.03,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Create Erdős-Rényi random graph.
    
    Each edge exists independently with probability p.
    Models random/uniform connections.
    
    Args:
        n: Number of nodes
        p: Edge probability
        seed: Random seed for reproducibility
    
    Returns:
        NetworkX Graph
    """
    return nx.erdos_renyi_graph(n, p, seed=seed)


def create_watts_strogatz(
    n: int = 200,
    k: int = 6,
    p: float = 0.1,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Create Watts-Strogatz small-world graph.
    
    Starts with a ring lattice and rewires edges with probability p.
    Exhibits high clustering and short average path length.
    
    Args:
        n: Number of nodes
        k: Each node connected to k nearest neighbors in ring topology
        p: Probability of rewiring each edge
        seed: Random seed for reproducibility
    
    Returns:
        NetworkX Graph
    """
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def create_powerlaw_cluster(
    n: int = 200,
    m: int = 3,
    p: float = 0.1,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Create powerlaw cluster graph.
    
    BA variant that adds triangle-forming connections, resulting
    in higher clustering than standard BA graphs.
    
    Args:
        n: Number of nodes
        m: Number of random edges to add for each new node
        p: Probability of adding a triangle after adding edge
        seed: Random seed for reproducibility
    
    Returns:
        NetworkX Graph
    """
    return nx.powerlaw_cluster_graph(n, m, p, seed=seed)


def load_snap_karate_club() -> nx.Graph:
    """Load Zachary's Karate Club network.
    
    Classic social network of 34 members of a university karate club.
    78 edges representing friendships.
    
    Returns:
        NetworkX Graph
    """
    return nx.karate_club_graph()


def load_snap_dolphins() -> nx.Graph:
    """Load a small social network similar to dolphin social network.
    
    Uses Les Misérables character co-appearance network as a
    proxy (77 nodes, built into NetworkX).
    
    Returns:
        NetworkX Graph
    """
    try:
        return nx.les_miserables_graph()
    except Exception:
        # Fallback: create similar sized network
        return nx.barabasi_albert_graph(62, 5, seed=42)


def download_snap_dataset(
    name: str,
    cache_dir: str | Path = "data/snap",
) -> Optional[nx.Graph]:
    """Load a SNAP dataset from local cache.
    
    Supported datasets (must be downloaded manually from SNAP):
    - email-Eu-core: Email network (1005 nodes, 25571 edges)
    - ca-GrQc: Collaboration network (5242 nodes)
    - facebook: Facebook social circles (4039 nodes)
    
    Args:
        name: Dataset name (e.g., "email-Eu-core")
        cache_dir: Directory containing dataset files
    
    Returns:
        NetworkX Graph, or None if dataset not found
    
    Note:
        Download datasets from https://snap.stanford.edu/data/
        and place the edge list file in cache_dir.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = {
        "email-Eu-core": "email-Eu-core.txt",
        "ca-GrQc": "ca-GrQc.txt",
        "facebook": "facebook_combined.txt",
    }

    if name not in dataset_files:
        print(f"Unknown dataset: {name}")
        print(f"Available: {list(dataset_files.keys())}")
        return None

    file_path = cache_dir / dataset_files[name]

    if not file_path.exists():
        print(f"Dataset file not found: {file_path}")
        print(f"\nTo download:")
        print(f"1. Go to https://snap.stanford.edu/data/")
        print(f"2. Download {name} dataset")
        print(f"3. Extract and place {dataset_files[name]} in {cache_dir}/")
        return None

    # Load edge list
    G = nx.read_edgelist(str(file_path), nodetype=int, create_using=nx.Graph())

    # Relabel nodes to 0-indexed
    G = nx.convert_node_labels_to_integers(G, first_label=0)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Get largest connected component
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G, first_label=0)

    print(f"✓ Loaded {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def get_all_test_networks(include_snap: bool = False) -> Dict[str, nx.Graph]:
    """Get dictionary of all test networks.
    
    Creates a collection of synthetic networks at various sizes
    and optionally loads real-world SNAP datasets.
    
    Args:
        include_snap: Whether to attempt loading SNAP datasets
    
    Returns:
        Dict mapping network name to NetworkX Graph
    """
    networks = {}

    # Barabási-Albert networks (scale-free)
    networks["BA-100"] = create_barabasi_albert(100, 3, seed=42)
    networks["BA-200"] = create_barabasi_albert(200, 3, seed=42)
    networks["BA-500"] = create_barabasi_albert(500, 3, seed=42)

    # Erdős-Rényi networks (random)
    networks["ER-100"] = create_erdos_renyi(100, 0.05, seed=42)
    networks["ER-200"] = create_erdos_renyi(200, 0.03, seed=42)
    networks["ER-500"] = create_erdos_renyi(500, 0.012, seed=42)

    # Watts-Strogatz networks (small-world)
    networks["WS-100"] = create_watts_strogatz(100, 6, 0.1, seed=42)
    networks["WS-200"] = create_watts_strogatz(200, 6, 0.1, seed=42)
    networks["WS-500"] = create_watts_strogatz(500, 6, 0.1, seed=42)

    # Powerlaw cluster graph
    networks["PLC-200"] = create_powerlaw_cluster(200, 3, 0.1, seed=42)

    # Small real networks
    networks["Karate"] = load_snap_karate_club()

    # SNAP datasets (if available)
    if include_snap:
        email = download_snap_dataset("email-Eu-core")
        if email is not None:
            networks["Email-Eu"] = email

    print(f"\n✓ Loaded {len(networks)} test networks")
    return networks


def print_network_statistics(graph: nx.Graph, name: str = "Network") -> None:
    """Print comprehensive network statistics.
    
    Computes and displays key structural properties of the network.
    
    Args:
        graph: NetworkX Graph to analyze
        name: Name to display in output
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    density = nx.density(graph)

    degrees = dict(graph.degree())
    avg_degree = np.mean(list(degrees.values()))
    max_degree = max(degrees.values())

    try:
        avg_clustering = nx.average_clustering(graph)
    except Exception:
        avg_clustering = 0.0

    try:
        diameter = nx.diameter(graph) if nx.is_connected(graph) else -1
    except Exception:
        diameter = -1

    print(f"\n{name} Statistics:")
    print(f"  Nodes: {n}")
    print(f"  Edges: {m}")
    print(f"  Density: {density:.4f}")
    print(f"  Avg degree: {avg_degree:.2f}")
    print(f"  Max degree: {max_degree}")
    print(f"  Avg clustering: {avg_clustering:.4f}")
    if diameter > 0:
        print(f"  Diameter: {diameter}")


if __name__ == "__main__":
    print("=" * 70)
    print("INTERCEPT: Network Datasets")
    print("=" * 70)

    # Load all synthetic networks
    networks = get_all_test_networks(include_snap=False)

    # Print statistics for each
    for name, G in networks.items():
        print_network_statistics(G, name)

    print("\n" + "=" * 70)
    print(f"Total networks: {len(networks)}")
    print("=" * 70)

    # Test SNAP loading
    print("\n\nTesting SNAP dataset loading...")
    snap_g = download_snap_dataset("email-Eu-core")
    if snap_g:
        print_network_statistics(snap_g, "Email-Eu-core")

from logging import warning
import torch
from torch_geometric.data import Data, InMemoryDataset
import networkx as nx
import random


class SSSPDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        num_graphs,
        n_nodes_range=(5, 20),
        m=1,
        p=0.15,
        q=0.0,
        max_hops=None,
        transform=None,
        pre_transform=None,
    ):
        self.num_graphs = num_graphs
        self.n_nodes_range = n_nodes_range  # Must be defined before generating graphs
        self.max_hops = max_hops
        self.m = m
        self.p = p
        self.q = q
        super().__init__(root, transform, pre_transform, force_reload=True)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @torch.no_grad()
    def _generate_by_num_nodes(self, num_nodes):
        G = nx.extended_barabasi_albert_graph(num_nodes, self.m, self.p, self.q)
        source = random.choice(list(G.nodes()))
        return G, source

    @torch.no_grad()
    def _generate_by_max_hops(self, max_hops, num_nodes):
        assert (
            max_hops < num_nodes
        ), f"max_hops {max_hops} should be smaller than num_nodes {num_nodes}"
        if max_hops > num_nodes / 2:
            warning(
                "max_hops larger than num_nodes/2, increasing num_nodes is recommended"
            )

        def get_max_hops(g, s):
            lengths = nx.single_source_shortest_path_length(g, s)
            mh = max(lengths.values())
            return mh

        G = nx.extended_barabasi_albert_graph(num_nodes, self.m, self.p, self.q)
        source = random.choice(list(G.nodes()))
        hops = get_max_hops(G, source)
        cnt = 1
        agg_hops = hops
        UPDATE_SOURCE_TIME = 5
        while hops != max_hops:
            if cnt % 100 == 0:
                warning(f"Tried {cnt} times to generate graph with specified max_hops")
            if cnt % UPDATE_SOURCE_TIME == 0:
                # if agg_hops / UPDATE_SOURCE_TIME < max_hops:
                #     # increase rewiring prob
                #     p -= min(0.05, p/2)
                # else:
                #     p += min(0.05, (1-p)/2)
                agg_hops = 0
                G = nx.extended_barabasi_albert_graph(num_nodes, self.m, self.p, self.q)
                source = random.choice(list(G.nodes()))
            else:
                source = random.choice(list(G.nodes()))
            hops = get_max_hops(G, source)
            agg_hops += hops
            cnt += 1
        return G, source

    def process(self):
        data_list = []
        print("NUM_GRAPHS:", self.num_graphs)
        for i in range(self.num_graphs):
            # --- Generate a random connected graph ---
            num_nodes = random.randint(*self.n_nodes_range)
            # Watts–Strogatz graphs require k to be even.
            if self.max_hops is None:
                G, source = self._generate_by_num_nodes(num_nodes)
            else:
                G, source = self._generate_by_max_hops(self.max_hops, num_nodes)

            # --- Assign random weights to edges ---
            for u, v in G.edges():
                G[u][v]["weight"] = random.random()

            # --- Choose a random source node ---
            # source = random.choice(list(G.nodes()))

            # --- Compute shortest path distances from the source ---
            path_lengths = nx.single_source_dijkstra_path_length(G, source)
            y = [path_lengths.get(i, float("inf")) for i in range(num_nodes)]

            # --- Create node features ---
            # Here, the only node feature is the binary source flag.
            x = torch.zeros((num_nodes, 1))
            x[source] = 1.0

            # --- Create edge_index and edge_attr ---
            edge_index = []
            edge_attr = []
            for u, v, data in G.edges(data=True):
                weight = data["weight"]
                # For undirected graphs, add both directions.
                edge_index.append([u, v])
                edge_index.append([v, u])
                edge_attr.append([weight])
                edge_attr.append([weight])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            # --- Build the Data object ---
            data_obj = Data(
                x=x,  # Node features (source flag)
                edge_index=edge_index,  # Graph connectivity
                edge_attr=edge_attr,  # Edge weights
                y=torch.tensor(y, dtype=torch.float),  # Ground-truth distances
            )
            data_list.append(data_obj)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = SSSPDataset(root="sssp_dataset", num_graphs=100)
    print(f"Dataset size: {len(dataset)}")
    print(dataset[0])

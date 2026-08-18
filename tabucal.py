"""Fast TabuCol implementation for undirected graph coloring.

The fixed-k search follows the improved TabuCol scheme: only conflicting
vertices are moved, move costs are read from an incremental vertex/color
conflict matrix, inverse moves are tabu, and a tabu move is accepted when it
improves the best conflict count seen in the current run.

The outer minimization first obtains a feasible DSATUR coloring.  It keeps that
coloring as the incumbent and invokes TabuCol only for smaller values of k.
This is important: TabuCol is a fixed-k local search, not a useful way to walk
down from an n-color solution one color at a time.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import snap

GRAPH_PATH = "out/graph.txt"
TEMPLATE_GRAPH_PATH = "src/graph.txt"


class TabuCol:
    """Color an undirected graph using DSATUR followed by TabuCol.

    For a fixed color count ``k``, ``neighbor_color_count[v][c]`` stores the
    number of neighbors of ``v`` currently using color ``c``.  Consequently a
    move delta is an O(1) lookup and applying a move only touches the moved
    vertex's neighbors.  One iteration costs O(|U| * k + degree(v)), where U is
    the set of currently conflicting vertices, instead of rescanning each
    candidate vertex's adjacency list for every possible color.
    """

    def __init__(
        self,
        node_num: int = 5,
        edge_prob: float = 0.5,
        random_seed: int = 42,
        max_iterations: int = 10000,
        max_restarts: int = 20,
        tabu_tenure_base: int = 10,
        tabu_tenure_alpha: float = 0.6,
    ):
        # Keep the historical ability to construct a small random graph, but
        # do not write it to disk.  Benchmark callers immediately load another
        # graph, so constructor I/O was pure overhead and an unexpected side
        # effect.
        self.graph = snap.GenRndGnm(
            snap.TUNGraph,
            node_num,
            int(edge_prob * node_num * (node_num - 1) / 2),
        )

        self.node_num = node_num
        self.edge_prob = edge_prob
        self.random_seed = random_seed
        self.max_iterations = max_iterations
        self.max_restarts = max_restarts
        # Galinier--Hao use Random(10) + 0.6 * number of conflicting
        # vertices.  Some later TabuCol variants scale by conflicting edges;
        # this implementation follows the former paper directly.  ``base``
        # is the exclusive upper bound of the random part.
        self.tabu_tenure_base = tabu_tenure_base
        self.tabu_tenure_alpha = tabu_tenure_alpha
        self.rng = random.Random(random_seed)

        self.node_ids: List[int] = []
        self.adj: Dict[int, Set[int]] = {}
        self.edges: List[Tuple[int, int]] = []
        self.node_color: Dict[int, int] = {}
        self.best_coloring: Dict[int, int] = {}
        self.best_color_count: int = 0
        self.best_conflicts: int = 10**9

        # Dense, zero-based views used only by the hot search loop.
        self._id_to_index: Dict[int, int] = {}
        self._adj_indices: List[List[int]] = []
        self._adj_index_sets: List[Set[int]] = []
        self._indexed_edges: List[Tuple[int, int]] = []
        self._last_run_iterations = 0
        self.last_search_stats: Dict[str, int] = {}

        self._rebuild_state()

    def _rebuild_state(self) -> None:
        self.node_ids = sorted(node.GetId() for node in self.graph.Nodes())
        self.node_num = len(self.node_ids)

        self.adj = {node_id: set() for node_id in self.node_ids}
        edge_set: Set[Tuple[int, int]] = set()
        for edge in self.graph.Edges():
            u = edge.GetSrcNId()
            v = edge.GetDstNId()
            if u == v:
                continue
            self.adj.setdefault(u, set()).add(v)
            self.adj.setdefault(v, set()).add(u)
            edge_set.add((u, v) if u < v else (v, u))

        self.edges = sorted(edge_set)
        self._id_to_index = {
            node_id: index for index, node_id in enumerate(self.node_ids)
        }
        self._adj_indices = [
            [self._id_to_index[neighbor] for neighbor in self.adj[node_id]]
            for node_id in self.node_ids
        ]
        self._adj_index_sets = [set(neighbors) for neighbors in self._adj_indices]
        self._indexed_edges = [
            (self._id_to_index[u], self._id_to_index[v]) for u, v in self.edges
        ]

        self.node_color = {node_id: 1 for node_id in self.node_ids}
        self.best_coloring = {}
        self.best_color_count = 0
        self.best_conflicts = len(self.edges)
        self.last_search_stats = {}

    def get_neighbors(self, node_id: int) -> List[int]:
        if node_id not in self.adj:
            print(f"Warning: Node {node_id} does not exist in the graph.")
            return []
        return sorted(self.adj[node_id])

    def _arrays_to_coloring(self, colors: List[int]) -> Dict[int, int]:
        return {
            node_id: colors[index] + 1
            for index, node_id in enumerate(self.node_ids)
        }

    def _random_greedy_colors(self, color_count: int) -> List[int]:
        """Build a randomized first-fit coloring capped at ``color_count``."""
        order = list(range(self.node_num))
        self.rng.shuffle(order)
        colors = [-1] * self.node_num

        for vertex in order:
            forbidden = [False] * color_count
            for neighbor in self._adj_indices[vertex]:
                neighbor_color = colors[neighbor]
                if neighbor_color >= 0:
                    forbidden[neighbor_color] = True

            new_color = -1
            for color in range(color_count):
                if not forbidden[color]:
                    new_color = color
                    break
            if new_color < 0:
                new_color = self.rng.randrange(color_count)
            colors[vertex] = new_color

        return colors

    def _project_coloring(
        self,
        coloring: Dict[int, int],
        color_count: int,
    ) -> List[int]:
        """Project a complete coloring into at most ``color_count`` classes.

        The largest classes are preserved.  Vertices in removed classes are
        reinserted into a minimum-conflict remaining class, with random tie
        breaking.  When reducing a feasible (k+1)-coloring this usually starts
        TabuCol much closer to feasibility than an unrelated random coloring.
        """
        if set(coloring) != set(self.node_ids):
            raise ValueError("initial_coloring must assign every graph vertex.")
        if any(not isinstance(coloring[node_id], int) for node_id in self.node_ids):
            raise ValueError("initial_coloring colors must be integers.")

        classes: Dict[int, List[int]] = {}
        for index, node_id in enumerate(self.node_ids):
            color = coloring[node_id]
            if color < 0:
                raise ValueError("initial_coloring colors must be non-negative.")
            classes.setdefault(color, []).append(index)

        ranked_classes = sorted(
            classes.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        kept = ranked_classes[:color_count]
        removed = ranked_classes[color_count:]
        colors = [-1] * self.node_num

        for new_color, (_, vertices) in enumerate(kept):
            for vertex in vertices:
                colors[vertex] = new_color

        displaced = [vertex for _, vertices in removed for vertex in vertices]
        self.rng.shuffle(displaced)
        for vertex in displaced:
            counts = [0] * color_count
            for neighbor in self._adj_indices[vertex]:
                neighbor_color = colors[neighbor]
                if neighbor_color >= 0:
                    counts[neighbor_color] += 1

            minimum = min(counts)
            chosen = -1
            ties = 0
            for color, count in enumerate(counts):
                if count == minimum:
                    ties += 1
                    if self.rng.randrange(ties) == 0:
                        chosen = color
            colors[vertex] = chosen

        return colors

    def _random_coloring(self, color_count: int) -> Dict[int, int]:
        return self._arrays_to_coloring(
            [self.rng.randrange(color_count) for _ in range(self.node_num)]
        )

    def _initial_coloring(self, color_count: int) -> Dict[int, int]:
        if color_count >= self.node_num:
            return {
                node_id: index + 1
                for index, node_id in enumerate(self.node_ids)
            }
        return self._arrays_to_coloring(self._random_greedy_colors(color_count))

    def count_conflicting_edges(
        self,
        coloring: Optional[Dict[int, int]] = None,
    ) -> int:
        if coloring is None:
            coloring = self.node_color
        return sum(
            1
            for u, v in self.edges
            if u in coloring and v in coloring and coloring[u] == coloring[v]
        )

    def _conflicting_vertices(self, coloring: Dict[int, int]) -> Set[int]:
        conflicting: Set[int] = set()
        for u, v in self.edges:
            if coloring[u] == coloring[v]:
                conflicting.add(u)
                conflicting.add(v)
        return conflicting

    def _move_delta(
        self,
        coloring: Dict[int, int],
        node_id: int,
        new_color: int,
    ) -> int:
        old_color = coloring[node_id]
        old_conflicts = 0
        new_conflicts = 0
        for neighbor in self.adj[node_id]:
            neighbor_color = coloring[neighbor]
            if neighbor_color == old_color:
                old_conflicts += 1
            if neighbor_color == new_color:
                new_conflicts += 1
        return new_conflicts - old_conflicts

    def _tabu_tenure(self, conflicting_vertex_count: int) -> int:
        random_part = (
            self.rng.randrange(self.tabu_tenure_base)
            if self.tabu_tenure_base > 0
            else 0
        )
        return int(self.tabu_tenure_alpha * conflicting_vertex_count) + random_part

    def _select_move_fast(
        self,
        colors: List[int],
        color_count: int,
        conflict_count: int,
        best_conflicts: int,
        tabu_until: List[List[int]],
        neighbor_color_count: List[List[int]],
        conflicting_vertices: List[int],
        iteration: int,
    ) -> Optional[Tuple[int, int, int]]:
        """Return a best admissible move, randomly breaking exact ties."""
        best_vertex = -1
        best_color = -1
        best_delta = 10**9
        tie_count = 0
        randrange = self.rng.randrange

        for vertex in conflicting_vertices:
            old_color = colors[vertex]
            counts = neighbor_color_count[vertex]
            old_conflicts = counts[old_color]
            tabu_row = tabu_until[vertex]

            for new_color in range(color_count):
                if new_color == old_color:
                    continue
                delta = counts[new_color] - old_conflicts
                if delta > best_delta:
                    continue

                projected = conflict_count + delta
                is_tabu = tabu_row[new_color] >= iteration
                if is_tabu and projected >= best_conflicts:
                    continue

                if delta < best_delta:
                    best_vertex = vertex
                    best_color = new_color
                    best_delta = delta
                    tie_count = 1
                else:
                    tie_count += 1
                    if randrange(tie_count) == 0:
                        best_vertex = vertex
                        best_color = new_color

        if best_vertex < 0:
            return None
        return best_vertex, best_color, best_delta

    def _run_tabucol(
        self,
        color_count: int,
        max_iterations: int,
        initial_coloring: Optional[Dict[int, int]],
        verbose: bool,
    ) -> Tuple[Optional[Dict[int, int]], int]:
        """Dense-array implementation of one fixed-k TabuCol run."""
        if initial_coloring is None:
            colors = self._random_greedy_colors(color_count)
        else:
            colors = self._project_coloring(initial_coloring, color_count)

        n = self.node_num
        neighbor_color_count = [[0] * color_count for _ in range(n)]
        tabu_until = [[0] * color_count for _ in range(n)]
        conflict_count = 0

        for u, v in self._indexed_edges:
            color_u = colors[u]
            color_v = colors[v]
            neighbor_color_count[u][color_v] += 1
            neighbor_color_count[v][color_u] += 1
            if color_u == color_v:
                conflict_count += 1

        conflicting_vertices: List[int] = []
        conflict_position = [-1] * n
        for vertex in range(n):
            if neighbor_color_count[vertex][colors[vertex]] > 0:
                conflict_position[vertex] = len(conflicting_vertices)
                conflicting_vertices.append(vertex)

        best_conflicts = conflict_count
        self._last_run_iterations = 0
        if conflict_count == 0:
            return self._compact_coloring(self._arrays_to_coloring(colors)), 0

        if color_count == 1:
            return None, conflict_count

        rng = self.rng
        fallback_moves = 0
        for iteration in range(1, max_iterations + 1):
            selected = self._select_move_fast(
                colors=colors,
                color_count=color_count,
                conflict_count=conflict_count,
                best_conflicts=best_conflicts,
                tabu_until=tabu_until,
                neighbor_color_count=neighbor_color_count,
                conflicting_vertices=conflicting_vertices,
                iteration=iteration,
            )

            # The standard aspiration rule can occasionally leave every move
            # tabu.  Use one explicit liveness move, as in Lewis's reference
            # implementation, instead of building a fallback list every round.
            if selected is None:
                fallback_moves += 1
                vertex = rng.choice(conflicting_vertices)
                old_color = colors[vertex]
                new_color = rng.randrange(color_count - 1)
                if new_color >= old_color:
                    new_color += 1
                counts = neighbor_color_count[vertex]
                selected = (
                    vertex,
                    new_color,
                    counts[new_color] - counts[old_color],
                )

            vertex, new_color, delta = selected
            old_color = colors[vertex]
            counts_vertex = neighbor_color_count[vertex]

            # Update membership of the moved vertex.  Its row in C does not
            # change because none of its neighbors changed color.
            if counts_vertex[old_color] > 0 and counts_vertex[new_color] == 0:
                position = conflict_position[vertex]
                last_vertex = conflicting_vertices.pop()
                if position < len(conflicting_vertices):
                    conflicting_vertices[position] = last_vertex
                    conflict_position[last_vertex] = position
                conflict_position[vertex] = -1
            elif counts_vertex[old_color] == 0 and counts_vertex[new_color] > 0:
                conflict_position[vertex] = len(conflicting_vertices)
                conflicting_vertices.append(vertex)

            colors[vertex] = new_color

            # Only neighbors of the moved vertex need their C rows updated.
            for neighbor in self._adj_indices[vertex]:
                counts_neighbor = neighbor_color_count[neighbor]
                neighbor_color = colors[neighbor]
                counts_neighbor[old_color] -= 1
                counts_neighbor[new_color] += 1

                if neighbor_color == old_color and counts_neighbor[old_color] == 0:
                    position = conflict_position[neighbor]
                    if position >= 0:
                        last_vertex = conflicting_vertices.pop()
                        if position < len(conflicting_vertices):
                            conflicting_vertices[position] = last_vertex
                            conflict_position[last_vertex] = position
                        conflict_position[neighbor] = -1
                elif neighbor_color == new_color and counts_neighbor[new_color] == 1:
                    if conflict_position[neighbor] < 0:
                        conflict_position[neighbor] = len(conflicting_vertices)
                        conflicting_vertices.append(neighbor)

            conflict_count += delta
            tenure = self._tabu_tenure(len(conflicting_vertices))
            tabu_until[vertex][old_color] = iteration + tenure
            self._last_run_iterations = iteration

            if conflict_count < best_conflicts:
                best_conflicts = conflict_count
                if verbose:
                    print(
                        f"[k={color_count}] iteration={iteration}, "
                        f"best_conflicts={best_conflicts}"
                    )
                if best_conflicts == 0:
                    self.last_search_stats["fallback_moves"] = (
                        self.last_search_stats.get("fallback_moves", 0)
                        + fallback_moves
                    )
                    coloring = self._arrays_to_coloring(colors)
                    return self._compact_coloring(coloring), 0

        self.last_search_stats["fallback_moves"] = (
            self.last_search_stats.get("fallback_moves", 0) + fallback_moves
        )
        return None, best_conflicts

    def tabucol(
        self,
        color_count: int,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
        initial_coloring: Optional[Dict[int, int]] = None,
    ) -> Tuple[Optional[Dict[int, int]], int]:
        """Run TabuCol for a fixed number of colors.

        Return ``(coloring, 0)`` on success.  On budget exhaustion return
        ``(None, best_conflict_count)``; failure is not a proof that the graph
        is not k-colorable.
        """
        if color_count <= 0:
            raise ValueError("color_count must be positive.")
        if max_iterations is None:
            max_iterations = self.max_iterations
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative.")
        if not self.node_ids:
            self._last_run_iterations = 0
            return {}, 0
        if color_count >= self.node_num:
            self._last_run_iterations = 0
            coloring = {
                node_id: index + 1
                for index, node_id in enumerate(self.node_ids)
            }
            return coloring, 0
        return self._run_tabucol(
            color_count=color_count,
            max_iterations=max_iterations,
            initial_coloring=initial_coloring,
            verbose=verbose,
        )

    def solve_fixed_k(
        self,
        color_count: int,
        max_iterations: Optional[int] = None,
        max_restarts: Optional[int] = None,
        verbose: bool = False,
        initial_coloring: Optional[Dict[int, int]] = None,
    ) -> Tuple[Optional[Dict[int, int]], int]:
        if max_restarts is None:
            max_restarts = self.max_restarts
        if max_restarts <= 0:
            raise ValueError("max_restarts must be positive.")

        # Repeating a one-color run cannot change anything: with at least one
        # edge its conflict count is fixed, and without edges it is feasible.
        if color_count == 1:
            self._last_run_iterations = 0
            if self.edges:
                return None, len(self.edges)
            coloring = {node_id: 1 for node_id in self.node_ids}
            return coloring, 0

        best_conflicts = 10**9
        total_iterations = 0
        for restart in range(1, max_restarts + 1):
            coloring, conflicts = self.tabucol(
                color_count=color_count,
                max_iterations=max_iterations,
                verbose=verbose,
                initial_coloring=initial_coloring if restart == 1 else None,
            )
            total_iterations += self._last_run_iterations
            if coloring is not None:
                self.last_search_stats["iterations"] = (
                    self.last_search_stats.get("iterations", 0)
                    + total_iterations
                )
                self.last_search_stats["restarts"] = (
                    self.last_search_stats.get("restarts", 0) + restart
                )
                if verbose:
                    print(f"[k={color_count}] feasible coloring at restart {restart}")
                return coloring, 0
            best_conflicts = min(best_conflicts, conflicts)

        self.last_search_stats["iterations"] = (
            self.last_search_stats.get("iterations", 0) + total_iterations
        )
        self.last_search_stats["restarts"] = (
            self.last_search_stats.get("restarts", 0) + max_restarts
        )
        return None, best_conflicts

    def _dsatur_coloring(self) -> Tuple[Dict[int, int], int]:
        """Compute the same deterministic DSATUR upper bound as dsatur.py."""
        if not self.node_ids:
            return {}, 0

        colors = [-1] * self.node_num
        adjacent_colors = [set() for _ in range(self.node_num)]
        uncolored = set(range(self.node_num))
        degrees = [len(neighbors) for neighbors in self._adj_indices]

        while uncolored:
            vertex = max(
                uncolored,
                key=lambda candidate: (
                    len(adjacent_colors[candidate]),
                    degrees[candidate],
                    -self.node_ids[candidate],
                ),
            )
            forbidden = adjacent_colors[vertex]
            color = 0
            while color in forbidden:
                color += 1
            colors[vertex] = color
            uncolored.remove(vertex)

            for neighbor in self._adj_indices[vertex]:
                if colors[neighbor] < 0:
                    adjacent_colors[neighbor].add(color)

        coloring = self._arrays_to_coloring(colors)
        return coloring, max(colors) + 1

    def _greedy_clique_lower_bound(self) -> int:
        """Return a cheap, certified clique lower bound on the color count."""
        if not self.node_ids:
            return 0
        if not self.edges:
            return 1

        order = sorted(
            range(self.node_num),
            key=lambda vertex: (-len(self._adj_indices[vertex]), self.node_ids[vertex]),
        )
        rank = [0] * self.node_num
        for position, vertex in enumerate(order):
            rank[vertex] = position

        best = 2
        for seed in order:
            if len(self._adj_indices[seed]) + 1 <= best:
                break
            candidates = {
                vertex
                for vertex in self._adj_index_sets[seed]
                if len(self._adj_indices[vertex]) + 1 > best
            }
            clique_size = 1
            while candidates:
                vertex = min(candidates, key=rank.__getitem__)
                clique_size += 1
                candidates.intersection_update(self._adj_index_sets[vertex])
            if clique_size > best:
                best = clique_size
        return best

    def solve(
        self,
        max_colors: Optional[int] = None,
        min_colors: int = 1,
        max_iterations: Optional[int] = None,
        max_restarts: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[Dict[int, int], int]:
        """Find the best coloring observed while successively reducing k.

        DSATUR supplies a feasible incumbent.  If its color count is within
        ``max_colors``, the first TabuCol target is one color fewer; a failed
        local search therefore never discards the known DSATUR solution.  If
        an explicit ``max_colors`` is below that incumbent and the capped
        search fails, the method returns the best known DSATUR coloring; the
        parameter is a search target, not a feasibility guarantee.
        """
        if min_colors <= 0:
            raise ValueError("min_colors must be positive.")
        if not self.node_ids:
            self.node_color = {}
            self.best_coloring = {}
            self.best_color_count = 0
            self.best_conflicts = 0
            self.last_search_stats = {}
            return {}, 0

        upper_limit = self.node_num if max_colors is None else min(
            max_colors, self.node_num
        )
        if upper_limit < min_colors:
            raise ValueError("max_colors must be greater than or equal to min_colors.")

        self.rng = random.Random(self.random_seed)
        self.last_search_stats = {
            "iterations": 0,
            "restarts": 0,
            "fixed_k_attempts": 0,
            "fallback_moves": 0,
        }
        dsatur_coloring, dsatur_count = self._dsatur_coloring()
        clique_lower_bound = self._greedy_clique_lower_bound()
        self.last_search_stats["dsatur_upper_bound"] = dsatur_count
        self.last_search_stats["clique_lower_bound"] = clique_lower_bound

        if dsatur_count <= upper_limit:
            best_coloring = dsatur_coloring
            best_color_count = dsatur_count
            target = min(upper_limit, dsatur_count - 1)
        else:
            # No feasible solution respecting the requested starting cap is
            # known yet.  Search at the cap, but retain DSATUR as a safe
            # fallback rather than returning the old n-color fallback.
            best_coloring = dsatur_coloring
            best_color_count = dsatur_count
            target = upper_limit

        last_failure_conflicts = 0
        effective_min = max(min_colors, clique_lower_bound)
        while target >= effective_min:
            self.last_search_stats["fixed_k_attempts"] += 1
            coloring, conflicts = self.solve_fixed_k(
                color_count=target,
                max_iterations=max_iterations,
                max_restarts=max_restarts,
                verbose=verbose,
                initial_coloring=best_coloring,
            )
            if coloring is None:
                last_failure_conflicts = conflicts
                if verbose:
                    print(
                        f"[k={target}] no feasible coloring found, "
                        f"best_conflicts={conflicts}"
                    )
                break

            best_coloring = coloring
            best_color_count = len(set(coloring.values()))
            if verbose:
                print(f"[k={target}] accepted, used_colors={best_color_count}")
            target = min(target - 1, best_color_count - 1)

        self.node_color = dict(best_coloring)
        self.best_coloring = dict(best_coloring)
        self.best_color_count = best_color_count
        # Preserve the historical meaning of this public attribute: it records
        # the best conflict count at the first failed target k (zero if no
        # target failed).  The returned incumbent itself is always feasible.
        self.best_conflicts = last_failure_conflicts
        self.last_search_stats["last_failure_conflicts"] = last_failure_conflicts
        self.last_search_stats["result_colors"] = best_color_count
        return dict(best_coloring), best_color_count

    def color_graph(
        self,
        max_colors: Optional[int] = None,
        min_colors: int = 1,
        max_iterations: Optional[int] = None,
        max_restarts: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[Dict[int, int], int]:
        return self.solve(
            max_colors=max_colors,
            min_colors=min_colors,
            max_iterations=max_iterations,
            max_restarts=max_restarts,
            verbose=verbose,
        )

    def _compact_coloring(self, coloring: Dict[int, int]) -> Dict[int, int]:
        color_map = {
            color_id: index + 1
            for index, color_id in enumerate(sorted(set(coloring.values())))
        }
        return {
            node_id: color_map[coloring[node_id]]
            for node_id in sorted(coloring)
        }

    def is_valid_coloring(
        self,
        coloring: Optional[Dict[int, int]] = None,
    ) -> bool:
        if coloring is None:
            coloring = self.node_color
        if set(coloring) != set(self.node_ids):
            return False
        if any(not isinstance(color, int) or color <= 0 for color in coloring.values()):
            return False
        return self.count_conflicting_edges(coloring) == 0

    def used_colors(self) -> List[int]:
        return sorted(set(self.node_color.values()))

    def get_state(self) -> Dict[str, object]:
        return {
            "colors": dict(self.node_color),
            "used_colors": self.used_colors(),
            "color_count": len(self.used_colors()),
            "conflicting_edges": self.count_conflicting_edges(),
            "valid_coloring": self.is_valid_coloring(),
            "search_stats": dict(self.last_search_stats),
        }

    def print_result(self) -> None:
        if not self.node_color:
            print("No coloring result.")
            return
        print(f"colors used: {len(self.used_colors())}")
        print(f"conflicting edges: {self.count_conflicting_edges()}")
        for node_id in sorted(self.node_color):
            print(f"node {node_id}, color {self.node_color[node_id]}")

    def load_graph(self, file_path: str) -> None:
        self.graph = snap.LoadEdgeList(snap.PUNGraph, file_path, 0, 1)
        self._rebuild_state()
        print(f"load graph from {file_path}")

    def save_graph(self, file_path: str) -> None:
        snap.SaveEdgeList(self.graph, file_path, "Undirected Graph Saved")
        print(f"graph saved to {file_path}")


def import_template_graph():
    return snap.LoadEdgeList(snap.PUNGraph, TEMPLATE_GRAPH_PATH, 0, 1)


def draw_graph(graph) -> None:
    # Plotting is optional; keep heavy imports out of solver startup time.
    import matplotlib.pyplot as plt
    import networkx as nx

    nx_graph = nx.Graph()
    for edge in graph.Edges():
        nx_graph.add_edge(edge.GetSrcNId(), edge.GetDstNId())
    nx.draw(nx_graph, with_labels=True)
    plt.show()


TabuCal = TabuCol
tabucal = TabuCol


def test_class() -> None:
    agent = TabuCol(random_seed=7, max_iterations=5000, max_restarts=30)
    agent.load_graph(TEMPLATE_GRAPH_PATH)
    agent.solve(verbose=True)
    agent.print_result()


def main() -> None:
    test_class()


if __name__ == "__main__":
    main()

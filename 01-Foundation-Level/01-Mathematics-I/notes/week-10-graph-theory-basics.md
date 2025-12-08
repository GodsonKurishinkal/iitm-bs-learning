# Week 10: Graph Theory Basics

**Course**: BSMA1001 - Mathematics I
**Level**: Foundation

---

## Learning Objectives
- Understand graph representations (adjacency matrix and adjacency list)
- Master Breadth-First Search (BFS) algorithm
- Master Depth-First Search (DFS) algorithm
- Learn Directed Acyclic Graphs (DAGs) and topological sorting

---

## 1. Graph Representations

### 1.1 Theory

A **graph** consists of vertices (nodes) and edges (connections). Different representations offer trade-offs between memory and access time.

### 1.2 Mathematical Definition

A graph is defined as $G = (V, E)$ where:
- $V$ = set of vertices
- $E$ = set of edges $\subseteq V \times V$

### 1.3 Representation Types

| Representation | Description | Space Complexity | Edge Lookup |
|----------------|-------------|------------------|-------------|
| **Adjacency Matrix** | $A[i][j] = 1$ if edge exists between $i$ and $j$ | $O(V^2)$ | $O(1)$ |
| **Adjacency List** | Each vertex stores list of neighbors | $O(V + E)$ | $O(\text{degree})$ |

### 1.4 Supply Chain Application

**Retail Context**: Supply chain networks are naturally graphs - warehouses as nodes, transportation links as edges. Graph representation enables network analysis and optimization.

---

## 2. Breadth-First Search (BFS)

### 2.1 Theory

**BFS** explores all neighbors at the current depth before moving to the next level. It finds **shortest paths in unweighted graphs**.

### 2.2 Algorithm

1. Start at source vertex, mark visited
2. Add to queue
3. While queue not empty:
   - Dequeue vertex
   - Visit all unvisited neighbors
   - Enqueue them

### 2.3 Key Properties

| Property | Value |
|----------|-------|
| **Time Complexity** | $O(V + E)$ |
| **Space Complexity** | $O(V)$ |
| **Data Structure** | Queue (FIFO) |
| **Path Type** | Shortest path (unweighted) |

### 2.4 Supply Chain Application

**Retail Context**: BFS finds **minimum-hop routes** - fewest transfers between distribution centers, or shortest path in terms of number of legs in multi-modal transportation.

---

## 3. Depth-First Search (DFS)

### 3.1 Theory

**DFS** explores as far as possible along each branch before backtracking. It's useful for:
- Detecting cycles
- Topological sorting
- Finding connected components

### 3.2 Algorithm (Recursive)

1. Mark current vertex as visited
2. For each unvisited neighbor, recursively call DFS

### 3.3 Key Properties

| Property | Value |
|----------|-------|
| **Time Complexity** | $O(V + E)$ |
| **Space Complexity** | $O(V)$ (recursion stack) |
| **Data Structure** | Stack (explicit or call stack) |
| **Traversal Type** | Depth-first exploration |

### 3.4 Supply Chain Application

**Retail Context**: DFS helps:
- Detect **circular dependencies** in Bill of Materials (BOM)
- Identify **connected components** in supplier networks
- Explore all possible delivery paths

---

## 4. DAGs and Topological Sorting

### 4.1 Theory

A **Directed Acyclic Graph (DAG)** has:
- Directed edges
- No cycles

**Topological sort** orders vertices so all edges point forward.

### 4.2 Mathematical Definition

**Topological Order**: Linear ordering of vertices such that for every edge $(u, v)$, vertex $u$ comes before vertex $v$.

### 4.3 Kahn's Algorithm

1. Find vertices with no incoming edges (in-degree = 0)
2. Remove vertex and add to result
3. Update in-degrees of neighbors
4. Repeat until graph is empty

### 4.4 Key Properties

| Property | Description |
|----------|-------------|
| **Existence** | Only exists for DAGs (no cycles) |
| **Uniqueness** | May have multiple valid orderings |
| **Time Complexity** | $O(V + E)$ |

### 4.5 Supply Chain Application

**Retail Context**: Topological sorting schedules dependent tasks:
- Manufacturing steps that must follow a specific order
- Determining which products to produce first based on component dependencies
- Order fulfillment workflows

---

## Summary Table

| Concept | Definition | Key Property | Supply Chain Application |
|---------|------------|--------------|--------------------------|
| **Adjacency Matrix** | 2D array where $A[i][j] = 1$ if edge exists | $O(1)$ edge lookup, $O(V^2)$ space | Dense network representation |
| **Adjacency List** | List of neighbors per vertex | $O(V+E)$ space, efficient for sparse | Sparse warehouse networks |
| **BFS** | Level-by-level exploration | Shortest path (unweighted) | Minimum-hop routes |
| **DFS** | Depth-first exploration | Cycle detection | BOM dependency checking |
| **Topological Sort** | Linear ordering respecting edges | Only for DAGs | Production scheduling |

---

## Key Takeaways

1. **Graph representation choice** depends on graph density - matrices for dense, lists for sparse graphs
2. **BFS guarantees shortest paths** in unweighted graphs using level-order traversal
3. **DFS is ideal for cycle detection** and exploring all paths systematically
4. **Topological sorting** provides valid execution order for dependent tasks in DAGs

---

## Next Week Preview

Week 11 covers **Graph Algorithms**:
- Shortest path algorithms (Dijkstra, Bellman-Ford)
- Minimum spanning trees

---

*IIT Madras BS Degree in Data Science*

/*
 *	Adapted from https://github.com/cran/pcalg
 *
 */

#pragma once

#include <iterator>
#include <vector>

#include <armadillo>
#include <boost/graph/adjacency_list.hpp>

typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS> InternalUndirectedGraph;
typedef boost::graph_traits<InternalUndirectedGraph>::out_edge_iterator UndirectedOutEdgeIterator;
typedef boost::graph_traits<InternalUndirectedGraph>::edge_iterator UndirectedEdgeIterator;

class GaussIndependenceTest {
 public:
  explicit GaussIndependenceTest(double* correlation_matrix, int n)
      : correlation_matrix_(correlation_matrix, n, n, false, true) {}

  virtual double Test(int u, int v, const std::vector<int>& separation_set) const;

 private:
  arma::mat correlation_matrix_;
};

class GraphSkeleton {
 public:
  explicit GraphSkeleton(double* correlation_matrix, int n) : independence_test_(correlation_matrix, n), graph_(n) {}

  void AddEdge(int u, int v) { boost::add_edge(u, v, graph_); }

  void RemoveEdge(int u, int v) { boost::remove_edge(u, v, graph_); }

  bool HasEdge(int u, int v) const { return boost::edge(u, v, graph_).second; }

  int GetNodeCount() const { return boost::num_vertices(graph_); }

  int GetEdgeCount() const { return boost::num_edges(graph_); }

  int GetDegree(int u) const { return boost::out_degree(u, graph_); }

  std::set<int> GetNeighbors(int u) const;

  void CreateLevelZero(double threshold, std::vector<std::vector<double>>* p_max_values,
                       std::vector<std::vector<std::vector<int>>>* separation_sets, int num_rows, int n);

  void FitToLevel(int level, double threshold, std::vector<std::vector<double>>* p_max_values,
                  std::vector<std::vector<std::vector<int>>>* separation_sets, int num_rows, int n);

 private:
  GaussIndependenceTest independence_test_;
  InternalUndirectedGraph graph_;
};

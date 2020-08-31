/*
 *	Adapted from https://github.com/cran/pcalg
 *
 */

#include <algorithm>
#include <iterator>
#include <limits>

#include <math.h>
#include <omp.h>

#include <boost/dynamic_bitset.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/log1p.hpp>

#include "constraint_heterogeneous.hpp"

double GaussIndependenceTest::Test(int u, int v, const std::vector<int>& separation_set) const {
  arma::mat C_sub;
  arma::uvec ind(separation_set.size() + 2);
  ind(0) = u;
  ind(1) = v;
  int i, j;
  for (i = 0; i < separation_set.size(); ++i) ind(i + 2) = separation_set[i];
  C_sub = correlation_matrix_.submat(ind, ind);
  for (i = 0; i < C_sub.n_rows; ++i)
    for (j = 0; j < C_sub.n_cols; ++j)
      if ((boost::math::isnan)(C_sub(i, j))) return std::numeric_limits<double>::quiet_NaN();

  double r, absz;
  if (separation_set.empty()) {
    r = correlation_matrix_(u, v);
    return abs(0.5 * log(abs((1 + r) / (1 - r))));
  } else if (separation_set.size() == 1) {
    r = (C_sub(0, 1) - C_sub(0, 2) * C_sub(1, 2)) /
        sqrt((1 - C_sub(1, 2) * C_sub(1, 2)) * (1 - C_sub(0, 2) * C_sub(0, 2)));
    return fabs(0.5 * (log(fabs((1 + r))) - log(fabs(1 - r))));
  } else {
    arma::mat PM;
    pinv(PM, C_sub);
    r = PM(0, 1) / sqrt(PM(0, 0) * PM(1, 1));
    return abs(0.5 * log(abs((1 + r) / (1 - r))));
  }
}

std::set<int> GraphSkeleton::GetNeighbors(int u) const {
  std::set<int> neighbors;
  UndirectedOutEdgeIterator outIter, outLast;

  for (boost::tie(outIter, outLast) = boost::out_edges(u, graph_); outIter != outLast; outIter++)
    neighbors.insert(boost::target(*outIter, graph_));

  return neighbors;
}

void GraphSkeleton::CreateLevelZero(double threshold, std::vector<std::vector<double>>* p_max_values,
                                    std::vector<std::vector<std::vector<int>>>* separation_sets, int num_rows, int n) {
  std::vector<int> us, vs;
  us.reserve(GetEdgeCount());
  vs.reserve(GetEdgeCount());

  for (int i = 0; i < num_rows; ++i) {
    for (int j = i + 1; j < n; ++j) {
      us.emplace_back(i);
      vs.emplace_back(j);
    }
  }

  std::vector<int> add_edges(us.size(), 0);
#pragma omp parallel for
  for (int i = 0; i < us.size(); ++i) {
    (*p_max_values)[us[i]][vs[i]] = independence_test_.Test(us[i], vs[i], {});
    if ((*p_max_values)[us[i]][vs[i]] < threshold) {
      (*separation_sets)[us[i]][vs[i]].resize(0);
    } else {
      add_edges[i] = 1;
    }
  }

  for (int i = 0; i < us.size(); ++i) {
    if (add_edges[i] == 1) {
      AddEdge(us[i], vs[i]);
    }
  }
}

void GraphSkeleton::FitToLevel(int level, double threshold, std::vector<std::vector<double>>* p_max_values,
                               std::vector<std::vector<std::vector<int>>>* separation_sets, int num_rows, int n) {
  bool found = true;

  UndirectedEdgeIterator ei, eiLast;

  std::vector<int> u, v;
  u.reserve(GetEdgeCount());
  v.reserve(GetEdgeCount());

  for (int i = 0; i < num_rows; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (HasEdge(i, j) && std::max(GetDegree(i), GetDegree(j)) > level) {
        u.emplace_back(i);
        v.emplace_back(j);
      }
    }
  }

  std::vector<int> deleteEdges(u.size(), 0);
  arma::ivec localEdgeTests(u.size(), arma::fill::zeros);

  found = u.size() > 0;

#pragma omp parallel for
  for (std::size_t l = 0; l < u.size(); l++) {
    bool edgeDone = false;

    int k;
    UndirectedOutEdgeIterator outIter, outLast;
    std::vector<int> condSet(level);
    std::vector<std::vector<int>::iterator> si(level);

    if (GetDegree(u[l]) > level) {
      std::vector<int> neighbors(0);
      neighbors.reserve(GetDegree(u[l]) - 1);
      for (boost::tie(outIter, outLast) = boost::out_edges(u[l], graph_); outIter != outLast; outIter++)
        if (boost::target(*outIter, graph_) != v[l]) neighbors.push_back(boost::target(*outIter, graph_));

      for (std::size_t i = 0; i < level; ++i) si[i] = neighbors.begin() + i;

      do {
        for (std::size_t i = 0; i < level; ++i) condSet[i] = *(si[i]);

        double pval = independence_test_.Test(u[l], v[l], condSet);
        localEdgeTests(l)++;

        if ((boost::math::isnan)(pval)) pval = 1.;
        if (pval > (*p_max_values)[u[l]][v[l]]) (*p_max_values)[u[l]][v[l]] = pval;
        if (pval < threshold) {
          deleteEdges[l] = 1;
          (*separation_sets)[u[l]][v[l]].resize(condSet.size());
          for (std::size_t j = 0; j < condSet.size(); ++j) (*separation_sets)[u[l]][v[l]][j] = condSet[j] + 1;
          edgeDone = true;
          break;
        }

        for (k = level - 1; k >= 0 && si[k] == neighbors.begin() + (neighbors.size() - level + k); --k)
          ;
        if (k >= 0) {
          si[k]++;
          for (k++; k < level; ++k) si[k] = si[k - 1] + 1;
        }
      } while (k >= 0);
    }

    if (!edgeDone && GetDegree(v[l]) > level) {
      std::vector<int> neighbors(0);
      std::vector<int> commNeighbors(0);
      neighbors.reserve(GetDegree(v[l]) - 1);
      commNeighbors.reserve(GetDegree(v[l]) - 1);
      int a;
      for (boost::tie(outIter, outLast) = boost::out_edges(v[l], graph_); outIter != outLast; outIter++) {
        a = boost::target(*outIter, graph_);
        if (a != u[l]) {
          if (HasEdge(u[l], a))
            commNeighbors.push_back(a);
          else
            neighbors.push_back(a);
        }
      }

      int m = neighbors.size();
      neighbors.insert(neighbors.end(), commNeighbors.begin(), commNeighbors.end());

      if (m > 0) {
        for (std::size_t i = 0; i < level; ++i) si[i] = neighbors.begin() + i;

        do {
          for (std::size_t i = 0; i < level; ++i) condSet[i] = *(si[i]);

          double pval = independence_test_.Test(v[l], u[l], condSet);
          localEdgeTests(l)++;

          if ((boost::math::isnan)(pval)) pval = 1.;
          if (pval > (*p_max_values)[u[l]][v[l]]) (*p_max_values)[u[l]][v[l]] = pval;
          if (pval < threshold) {
            deleteEdges[l] = 1;

            (*separation_sets)[u[l]][v[l]].resize(condSet.size());
            for (std::size_t j = 0; j < condSet.size(); ++j) (*separation_sets)[u[l]][v[l]][j] = condSet[j] + 1;
            edgeDone = true;
            break;
          }

          for (k = level - 1; k >= 0 && si[k] == neighbors.begin() + (neighbors.size() - level + k); --k)
            ;

          if (k == 0 && si[0] == neighbors.begin() + (m - 1)) k = -1;
          if (k >= 0) {
            si[k]++;
            for (k++; k < level; ++k) si[k] = si[k - 1] + 1;
          }
        } while (k >= 0);
      }
    }
  }

  for (std::size_t l = 0; l < deleteEdges.size(); ++l) {
    if (deleteEdges[l] == 1) {
      RemoveEdge(u[l], v[l]);
    }
  }
}

// ======================================================================== //
// Copyright 2022-2023 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "cukd/helpers.h"

namespace cukd {

  /*! traverse k-d tree with periodic stack-based (sb) traversal */
  template<typename result_t,
           typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  void traverse_stack_based_periodic(result_t &result,
                        typename data_traits::point_t queryPoint,
                        const box_t<typename data_traits::point_t> d_bounds,
                        const data_t *d_nodes,
                        int numPoints,
                        const typename data_traits::point_t *periodic_box_size=nullptr)
  {
    using point_t  = typename data_traits::point_t;
    using point_traits = ::cukd::point_traits<point_t>;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };
    
    scalar_t cullDist = result.initialCullDist2();

    /* can do at most 2**30 points... */
    struct StackEntry {
      int   nodeID;
      float sqrDist;
      int splitDim;
      bool isLeft;
      float farSideBound;
    };
    StackEntry stackBase[30];
    StackEntry *stackPtr = stackBase;

    /*! current node in the tree we're traversing */
    int curr = 0;

    auto bounds = d_bounds;
    
    while (true) {
      while (curr < numPoints) {
        CUKD_STATS(if (cukd::g_traversalStats) ::atomicAdd(cukd::g_traversalStats,1));
        const data_t &curr_node  = d_nodes[curr];

        const int  curr_dim
          = data_traits::has_explicit_dim
          ? data_traits::get_dim(curr_node)
          : (BinaryTree::levelOf(curr) % num_dims);
        
        const auto sqrDist = sqrDistance(data_traits::get_point(curr_node), queryPoint, periodic_box_size);
        
        cullDist = result.processCandidate(curr,sqrDist);

        const auto node_coord   = data_traits::get_coord(curr_node,curr_dim);
        const auto query_coord  = get_coord(queryPoint,curr_dim);
        const bool  leftIsClose = query_coord < node_coord;
        const int   lChild = 2*curr+1;
        const int   rChild = lChild+1;

        const int closeChild = leftIsClose?lChild:rChild;
        const int farChild   = leftIsClose?rChild:lChild;
        
        float sqrDistToPlane = sqr(query_coord - node_coord);

        float farChildFarBound;
        if(leftIsClose)
          farChildFarBound = point_traits::get_coord(bounds.upper, curr_dim);
        else
          farChildFarBound = point_traits::get_coord(bounds.lower, curr_dim);
        
        // box lenght <= 0 means no periodicity in that dimension
        if(periodic_box_size != nullptr) {
          const scalar_t box_size =  point_traits::get_coord(*periodic_box_size, curr_dim);
          if(box_size > 0) {
            const float farChildFarBoundDist = abs(query_coord - farChildFarBound);
            const float farChildFarBoundSqrDist = sqr(min(farChildFarBoundDist, box_size - farChildFarBoundDist));
            sqrDistToPlane = min(sqrDistToPlane, farChildFarBoundSqrDist);
          }
        }

        if (sqrDistToPlane < cullDist && farChild < numPoints) {
          stackPtr->nodeID  = farChild;
          stackPtr->sqrDist = sqrDistToPlane;
          stackPtr->splitDim = curr_dim;
          stackPtr->isLeft = !leftIsClose;
          stackPtr->farSideBound = farChildFarBound;
          ++stackPtr;
        }
        if(leftIsClose)
          point_traits::set_coord(bounds.upper, curr_dim, node_coord);
        else
          point_traits::set_coord(bounds.lower, curr_dim, node_coord);
        curr = closeChild;
      }

      while (true) {
        if (stackPtr == stackBase) 
          return;
        --stackPtr;
        curr = stackPtr->nodeID;
        const auto splitDim = stackPtr->splitDim;
        const data_t &curr_node  = d_nodes[curr];
        const auto node_coord = data_traits::get_coord(curr_node, splitDim);
        if(stackPtr->isLeft) {
          point_traits::set_coord(bounds.upper, splitDim, node_coord);
          point_traits::set_coord(bounds.lower, splitDim, stackPtr->farSideBound);
        } else {
          point_traits::set_coord(bounds.upper, splitDim, stackPtr->farSideBound);
          point_traits::set_coord(bounds.lower, splitDim, node_coord);
        }
        if (stackPtr->sqrDist >= cullDist)
          continue;
        break;
      }
    }
  }
  
}

// ======================================================================== //
// Copyright 2018-2022 Ingo Wald                                            //
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

namespace cukd {

  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  int get_node_dim(int node_id, const data_t *d_nodes)
  {
    using point_t  = typename data_traits::point_t;
    enum { n_dim = num_dims_of<point_t>::value };

    const auto& node = d_nodes[node_id];
    const int expand_dim
        = data_traits::has_explicit_dim
        ? data_traits::get_dim(node)
        : (BinaryTree::levelOf(node_id) % n_dim);
    return expand_dim;
  }

  inline __device__ int parent_node(const int node) { return (node+1)/2-1; }

  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  box_t<typename data_traits::point_t>
  expand_bounds_to_parent(box_t<typename data_traits::point_t> bounds,
                          int current_node_id,
                          int parent_node_id,
                          const data_t *d_nodes,
                          box_t<typename data_traits::point_t> world_bounds)
  {
    using point_t  = typename data_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { n_dim = num_dims_of<point_t>::value };
    
    const int parent_expand_dim = get_node_dim<data_t, data_traits>(parent_node_id, d_nodes);
    const int current_node_is_left = BinaryTree::isLeftSibling(current_node_id);

    int super_node_child_id = parent_node_id;
    // Need to go to at least grandparrent to get bounds
    int super_node_id = parent_node(super_node_child_id);
    int super_node_expand_dim = get_node_dim<data_t, data_traits>(super_node_id, d_nodes);

    /* Move up the tree until either the root or a node on the same side
    and with the same direction as the current node is found */
    while(super_node_child_id > 0
          && (current_node_is_left != BinaryTree::isLeftSibling(super_node_child_id)
              || super_node_expand_dim != parent_expand_dim)) {
      super_node_child_id = super_node_id;
      super_node_id = parent_node(super_node_id);
      super_node_expand_dim = get_node_dim<data_t, data_traits>(super_node_id, d_nodes);
    }
    
    float expand_pos;
    if(super_node_id >= 0) {
      const auto& super_node = d_nodes[super_node_id];
      expand_pos = data_traits::get_coord(super_node, parent_expand_dim);
    } else {
      // Above root node, use world bounds
      if(current_node_is_left)
        // Expand right
        expand_pos = get_coord(world_bounds.upper, parent_expand_dim);
      else
        // Expand left
        expand_pos = get_coord(world_bounds.lower, parent_expand_dim);
    }

    if(current_node_is_left) {
      set_coord(bounds.upper, parent_expand_dim, expand_pos);
    } else {
      set_coord(bounds.lower, parent_expand_dim, expand_pos);
    }

    return bounds;
  }


  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  box_t<typename data_traits::point_t>
  shrink_bounds_to_child(box_t<typename data_traits::point_t> bounds,
                         int current_node_id,
                         int child_node_id,
                         const data_t *d_nodes)
  {
    const int shrink_dim = get_node_dim<data_t, data_traits>(current_node_id, d_nodes);
    const auto& current_node = d_nodes[current_node_id];
    const auto shrink_pos = data_traits::get_coord(current_node, shrink_dim);

    if(BinaryTree::isLeftSibling(child_node_id)) {
      // Left child, shrink upper bounds
      set_coord(bounds.upper, shrink_dim, shrink_pos);
    } else {
      // Right child, shrink lower bounds
      set_coord(bounds.lower, shrink_dim, shrink_pos);
    }

    return bounds;
  }


  template<typename point_t>
  inline __device__
  auto
  smallest_distance_to_bounds(const box_t<point_t>& bounds,
                              const point_t& point)
  {
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { n_dim = num_dims_of<point_t>::value };

    const point_t min_vector = min(point - bounds.lower, bounds.upper - point);

    scalar_t min_dist = INFINITY;
    for (int i = 0; i < n_dim; ++i) {
      scalar_t d = get_coord(min_vector, i);
      if (d >= min_dist) continue;
      min_dist = d;
    }
    return min_dist;
  }


  template<typename result_t,
           typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  void traverse_stack_free_bounds_tracking(result_t &result,
                       typename data_traits::point_t queryPoint,
                       const box_t<typename data_traits::point_t> worldBounds,
                       const data_t *d_nodes,
                       int numPoints,
                       const typename data_traits::point_t *periodic_box_size=nullptr)
  {
    using point_t  = typename data_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };

    float cullDist = result.initialCullDist2();
    
    int previous_node_id = -1;
    int current_node_id = 0;

    box_t<point_t> current_bounds = worldBounds;
    
    // printf("world bounds: (%f, %f), (%f, %f)\n",
    //   current_bounds.lower.x, current_bounds.upper.x,
    //   current_bounds.lower.y, current_bounds.upper.y);
    while (true) {
      const int parent_node_id = parent_node(current_node_id);
      
      if (current_node_id >= numPoints) {
        // Child does not exist, go back to parent
        previous_node_id = current_node_id;
        current_node_id = parent_node_id;
        continue;
      }

      CUKD_STATS(if (cukd::g_traversalStats) ::atomicAdd(cukd::g_traversalStats,1));

      const auto &current_node = d_nodes[current_node_id];
      const bool from_parent = (previous_node_id < current_node_id);
      if(from_parent) {
        const auto dist_sqr =
          sqrDistance(queryPoint, data_traits::get_point(current_node), periodic_box_size);
        cullDist = result.processCandidate(current_node_id, dist_sqr);
      }

      const int current_node_dim = get_node_dim<data_t, data_traits>(current_node_id, d_nodes);
      const float split_pos = data_traits::get_coord(current_node, current_node_dim);
      const float query_pos = get_coord(queryPoint, current_node_dim);

      const float dist_to_split = query_pos - split_pos;
      const int   close_side = dist_to_split > 0.f;
      const int   close_child_node_id = BinaryTree::leftChildOf(current_node_id) + close_side;
      const int   far_child_node_id   = BinaryTree::rightChildOf(current_node_id) - close_side;

      bool far_child_in_range = sqr(dist_to_split) <= cullDist;

      // If periodic dimensions exist and far child is not already in range,
      // check if far child bounds are in reach when wrapping around
      if(periodic_box_size != nullptr && !far_child_in_range) {
        const scalar_t box_size = get_coord(*periodic_box_size, current_node_dim);
        if(box_size > 0) {
          scalar_t d = 0;
          if(BinaryTree::isLeftSibling(far_child_node_id))
            d = abs(query_pos - get_coord(current_bounds.lower, current_node_dim));
          else
            d = abs(get_coord(current_bounds.upper, current_node_dim) - query_pos);
          
          far_child_in_range = sqr(min(d, box_size - d)) <= cullDist;
        }
      }

      // if (previous_node_id == close_child_node_id)
      //   // if we came from the close child, we may still have to check
      //   // the far side - but only if this exists, and if far half of
      //   // current space if even within search radius.
      //   next_node_id
      //     = ((far_child_node_id<numPoints) && far_child_in_range)
      //     ? far_child_node_id
      //     : parent_node_id;
      // else if (previous_node_id == far_child_node_id)
      //   // if we did come from the far child, then both children are
      //   // done, and we can only go up.
      //   next_node_id = parent_node_id;
      // else
      //   // we didn't come from any child, so must be coming from a
      //   // parent... we've already been processed ourselves just now,
      //   // so next stop is to look at the children (unless there
      //   // aren't any). this still leaves the case that we might have
      //   // a child, but only a far child, and this far child may or
      //   // may not be in range ... we'll fix that by just going to
      //   // near child _even if_ only the far child exists, and have
      //   // that child do a dummy traversal of that missing child, then
      //   // pick up on the far-child logic when we return.
      //   next_node_id
      //     = (2*current_node_id+1<numPoints)
      //     ? close_child_node_id
      //     : parent_node_id;
  
      int next_node_id = parent_node_id;
      if(from_parent) {
        if(close_child_node_id < numPoints)
          // Came from parent, go to close child if it exists
          next_node_id = close_child_node_id;
        else if(far_child_in_range && far_child_node_id < numPoints)
          next_node_id = far_child_node_id;
      }
      else if(previous_node_id == close_child_node_id) {
        if(far_child_in_range && far_child_node_id < numPoints)
          // Far child in range and exists
          next_node_id = far_child_node_id;
      }

      if(next_node_id == -1)
        // Arrived back at the root, so we are done
        return;
      
      // printf("query: (%f, %f), node: %d, split: %f, split_dim: %d, from_parent: %d, far_child_in_range: %d, next: %d, current bounds: (%f, %f), (%f, %f)\n",
      //          queryPoint.x, queryPoint.y,
      //          current_node_id,
      //          split_pos,
      //          current_node_dim,
      //          from_parent,
      //          far_child_in_range,
      //          next_node_id,
      //          current_bounds.lower.x, current_bounds.upper.x,
      //          current_bounds.lower.y, current_bounds.upper.y);
      // if(!current_bounds.contains(queryPoint))
      //   printf("Not in bounds!");
      // : query: (%f, %f), node: %d, split: %f, split_dim: %d, from_parent: %d, far_child_in_range: %d, next: %d, current bounds: (%f, %f), (%f, %f)\n",
      //           queryPoint.x, queryPoint.y,
      //           current_node_id,
      //           split_pos,
      //           current_node_dim,
      //           from_parent,
      //           far_child_in_range,
      //           next_node_id,
      //           current_bounds.lower.x, current_bounds.upper.x,
      //           current_bounds.lower.y, current_bounds.upper.y);

      if(next_node_id == parent_node_id) {
        // We are going back up the tree. Check if any of the current bounds
        // are within the cull distance, else we are done.
        if(current_bounds.contains(queryPoint) && sqr(smallest_distance_to_bounds(current_bounds, queryPoint)) > cullDist)
          // We finished the subtree and all current sides of the current bounds are out of reach, so no need to search other parts of the tree
          return;
        current_bounds = expand_bounds_to_parent<data_t, data_traits>(current_bounds, current_node_id, parent_node_id, d_nodes, worldBounds);
        // printf("expanded bounds to (%f, %f), (%f, %f)\n",
        //     current_bounds.lower.x, current_bounds.upper.x,
        //     current_bounds.lower.y, current_bounds.upper.y);
      } else {
        if(next_node_id < numPoints)
          // Shrink, unless we go to a child that does not exist (next_node_id >= N)
          current_bounds = shrink_bounds_to_child<data_t, data_traits>(current_bounds, current_node_id, next_node_id, d_nodes);
          // printf("shrunk bounds to (%f, %f), (%f, %f)\n",
          //   current_bounds.lower.x, current_bounds.upper.x,
          //   current_bounds.lower.y, current_bounds.upper.y);
      }

      previous_node_id = current_node_id;
      current_node_id = next_node_id;
    }
  }
}

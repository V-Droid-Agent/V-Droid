from typing import Any
from android_env.proto.a11y import android_accessibility_forest_pb2
from android_world.agents.m3a_utils import _logical_to_physical, get_color_for_group


def turn_tree_to_group_bounding_boxes(
    orientation: str,
    logical_screen_size: list,
    physical_frame_boundary: list,
    forest: android_accessibility_forest_pb2.AndroidAccessibilityForest | Any,
    exclude_invisible_elements: bool = True,
) -> dict:
    display_id_counter = 0
    extra_attributes = {}
    group_bounding_boxes = {}
    group_index = 0  # Counter to assign colors to each group

    for window in forest.windows:
        for node in window.tree.nodes:
            node_display_id = None
            if not node.child_ids or node.content_description or node.is_scrollable:
                if exclude_invisible_elements and not node.is_visible_to_user:
                    node_display_id = None
                else:
                    node_display_id = display_id_counter

            extra_attributes[(window.id, node.unique_id)] = {
                "display_id": node_display_id}

            if node_display_id is not None:
                display_id_counter += 1

    def format_node(node, window_id):
        """Recursively calculates bounding boxes for a node and its children."""
        bounding_boxes = []
        nonlocal group_index

        # Calculate bounding box for the node itself, if applicable
        if node.bounds_in_screen:
            upper_left_logical, lower_right_logical = _ui_element_logical_corner(
                node, orientation
            )
            upper_left_physical = _logical_to_physical(
                upper_left_logical,
                logical_screen_size,
                physical_frame_boundary,
                orientation,
            )
            lower_right_physical = _logical_to_physical(
                lower_right_logical,
                logical_screen_size,
                physical_frame_boundary,
                orientation,
            )
            bounding_boxes.append((upper_left_physical, lower_right_physical))

        # Recursively calculate bounding boxes for child nodes if they exist
        child_bounding_boxes = []
        if node.child_ids:
            for child_id in node.child_ids:
                # Find the child node by ID within the same window
                child_node = next(
                    (n for n in window.tree.nodes if n.unique_id == child_id), None)
                if child_node and (window_id, child_id) not in processed_nodes:
                    processed_nodes.add((window_id, child_id))
                    _, child_boxes = format_node(child_node, window_id)
                    # Add each child bounding box to the list for group calculation
                    child_bounding_boxes.extend(child_boxes)

            # Calculate the bounding box for the entire group of children
            if child_bounding_boxes:
                group_top_left = (
                    min(box[0][0] for box in child_bounding_boxes),
                    min(box[0][1] for box in child_bounding_boxes),
                )
                group_bottom_right = (
                    max(box[1][0] for box in child_bounding_boxes),
                    max(box[1][1] for box in child_bounding_boxes),
                )
                # Assign a color to this group based on group_index
                group_color = get_color_for_group(group_index)
                group_bounding_boxes[(window_id, node.unique_id)] = {
                    "bounding_box": (group_top_left, group_bottom_right),
                    "color": group_color
                }
                group_index += 1  # Increment the group index for the next group

        return None, bounding_boxes + child_bounding_boxes

    # Initialize the bounding box calculation
    for window in forest.windows:
        processed_nodes = set()  # Track processed nodes to avoid duplication

        for node in window.tree.nodes:
            if (window.id, node.unique_id) not in processed_nodes:
                processed_nodes.add((window.id, node.unique_id))
                format_node(node, window.id)

    return group_bounding_boxes


def _ui_element_logical_corner(
    node, orientation: int
) -> list[tuple[int, int]]:
    """Get logical coordinates for corners of a given protobuf node.

    Args:
        node: The protobuf node with a bounds_in_screen attribute.
        orientation: The current orientation.

    Returns:
        Logical coordinates for upper left and lower right corner for the node.

    Raises:
        ValueError: If bounding box is missing or orientation is invalid.
    """
    if not hasattr(node, 'bounds_in_screen') or node.bounds_in_screen is None:
        raise ValueError('Node does not have bounds_in_screen.')

    # Extract bounding box coordinates from bounds_in_screen
    x_min = getattr(node.bounds_in_screen, 'left', None)
    y_min = getattr(node.bounds_in_screen, 'top', None)
    x_max = getattr(node.bounds_in_screen, 'right', None)
    y_max = getattr(node.bounds_in_screen, 'bottom', None)

    if None in [x_min, y_min, x_max, y_max]:
        raise ValueError('Bounding box coordinates are incomplete.')

    # Return logical coordinates based on orientation
    if orientation == 0:
        return [(int(x_min), int(y_min)), (int(x_max), int(y_max))]
    elif orientation == 1:
        return [(int(x_min), int(y_max)), (int(x_max), int(y_min))]
    elif orientation == 2:
        return [(int(x_max), int(y_max)), (int(x_min), int(y_min))]
    elif orientation == 3:
        return [(int(x_max), int(y_min)), (int(x_min), int(y_max))]
    else:
        raise ValueError('Unsupported orientation.')

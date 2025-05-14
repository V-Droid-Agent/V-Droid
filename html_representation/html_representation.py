from typing import Any, Optional
from android_env.proto.a11y import android_accessibility_forest_pb2
import dataclasses
import pdb
from transformers import AutoTokenizer
import re
from bs4 import BeautifulSoup, NavigableString, Tag

volume_keywords = {"media volume", "call volume",
                   "ring & notification volume", "alarm volume"}


def turn_tree_to_clean_html_input(
    forest: android_accessibility_forest_pb2.AndroidAccessibilityForest | Any,
    exclude_invisible_elements: bool = True,
) -> str:

    html_desc = turn_tree_to_html_input(
        forest, exclude_invisible_elements=True)
    html_desc = aggregate_html_cleanup(html_desc)
    # html_desc = remove_empty_divs_keep_indent(html_desc)
    html_desc = custom_one_tag_per_line_no_selfclose(html_desc)
    # html_desc = finalize_html_attributes_keep_format(html_desc)
    return html_desc


def remove_empty_divs_keep_indent(html_str: str) -> str:
    """
    Removes empty <div> blocks from html_str but tries to preserve
    the original indentation/formatting of the remaining HTML.
    """
    pattern = r"<div>(?:\s|<br\s*/?>)*</div>"
    new_html = html_str
    while True:
        old_html = new_html
        new_html = re.sub(pattern, "", new_html)
        if new_html == old_html:
            break

    soup = BeautifulSoup(new_html, "html.parser")

    def recursively_remove_empty_divs(soup_element):
        for div in soup_element.find_all("div"):
            if not div.text.strip() and not div.find(True):
                div.decompose()

        return soup_element

    soup = recursively_remove_empty_divs(soup)
    final_html = str(soup)

    final_html = re.sub(r'\n\s*\n', '\n', final_html)

    return final_html


def custom_one_tag_per_line_no_selfclose(html_str: str, indent_spaces=2) -> str:
    soup = BeautifulSoup(html_str, "html.parser")
    merge_text_into_input(soup)
    lines = []
    for child in soup.contents:
        if isinstance(child, Tag):
            lines.extend(dfs_pretty_print_no_selfclose(
                child, 0, indent_spaces))
        elif isinstance(child, NavigableString):
            text = child.strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


VOID_ELEMENTS = {"area", "base", "br", "col", "embed", "hr", "img",
                 "link", "meta", "param", "source", "track", "wbr"}


def dfs_pretty_print_no_selfclose(elem, level=0, indent_spaces=2) -> list[str]:
    """
    Recursively formats `elem` with indentation. 
    - If the tag is in VOID_ELEMENTS (like <img>), use <tag ... /> self-closing
    - If zero children => <tag></tag> on one line
    - If exactly one text child => inline
    - Otherwise multi-line
    BUT if a text node follows a single-line tag, we put them on the same line.
    """
    lines = []
    indent = " " * (indent_spaces * level)

    # If it's just text
    if isinstance(elem, NavigableString):
        text = elem.strip()
        if text:
            lines.append(indent + text)
        return lines

    if not isinstance(elem, Tag):
        # e.g. Comment or Doctype
        return lines

    # Build the start tag text: <tag attr="...">
    start_tag_text = elem.name
    attr_strs = []
    for k, v in elem.attrs.items():
        if isinstance(v, list):
            v = " ".join(v)
        attr_strs.append(f'{k}="{v}"')
    if attr_strs:
        start_tag_text += " " + " ".join(attr_strs)

    # Gather children ignoring whitespace-only text
    children = []
    for c in elem.children:
        if isinstance(c, NavigableString) and not c.strip():
            continue
        children.append(c)

    # 1) Void element => self-close
    if elem.name.lower() in VOID_ELEMENTS:
        lines.append(f'{indent}<{start_tag_text} />')
        return lines

    # 2) Zero children => single-line <tag></tag>
    if len(children) == 0:
        lines.append(f'{indent}<{start_tag_text}></{elem.name}>')
        return lines

    # 3) Exactly one text child => inline
    if len(children) == 1 and isinstance(children[0], NavigableString):
        text_content = children[0].strip()
        lines.append(f'{indent}<{start_tag_text}>{text_content}</{elem.name}>')
        return lines

    # 4) Otherwise, multi-line approach
    #    We'll iterate children with an index so we can see if the next child is text
    lines.append(f'{indent}<{start_tag_text}>')

    i = 0
    while i < len(children):
        child = children[i]
        sub_lines = dfs_pretty_print_no_selfclose(
            child, level+1, indent_spaces)

        # If sub_lines is exactly one line, check next child
        if len(sub_lines) == 1 and i+1 < len(children):
            next_child = children[i+1]
            # Is the next child a text node? Then append to the same line
            if isinstance(next_child, NavigableString):
                txt = next_child.strip()
                if txt:
                    # Append it to the last line
                    sub_lines[-1] += f" {txt}"
                i += 1  # consume the next child, so it won't produce its own line

        # Add sub_lines
        lines.extend(sub_lines)
        i += 1

    lines.append(f'{indent}</{elem.name}>')
    return lines


def merge_text_into_input(soup: BeautifulSoup):
    """
    For each <input> with zero children, 
    if its next sibling is a text node (non-empty),
    we move that text inside the <input> tag as the node's string.
    Then remove that text node from the parent.
    """

    # Find all <input> tags
    all_inputs = soup.find_all("input")
    for inp in all_inputs:
        # If <input> already has children (like 'some text'), skip
        # Or if it's self-closing or has an attribute that indicates no text
        if any(child for child in inp.children if not (isinstance(child, NavigableString) and not child.strip())):
            continue

        # Check the next sibling
        next_sib = inp.next_sibling
        if next_sib and isinstance(next_sib, NavigableString):
            label_text = next_sib.strip()
            if label_text:
                # Move label_text into <input> as its content
                inp.string = (inp.string or "") + label_text
                # Remove the text node from the parent flow
                next_sib.extract()

    return soup


def finalize_html_attributes_keep_format(html_str: str) -> str:
    """
    1) Convert id="NNN"  -->  id=NNN   (for numeric IDs)
    2) Convert text="SOMETHING" --> text='SOMETHING'
    Preserves all original whitespace/indentation exactly.
    """

    new_html = re.sub(r'id="(\d+)"', r'id=\1', html_str)

    new_html = re.sub(r'text="([^"]*)"', r"text='\1'", new_html)

    return new_html


def turn_tree_to_html_input(
    forest: android_accessibility_forest_pb2.AndroidAccessibilityForest | Any,
    exclude_invisible_elements: bool = True,
) -> str:
    """Extracts nodes from accessibility forest and converts to nested HTML."""

    display_id_counter = 0
    extra_attributes = {}

    # action_counts = {
    #     "clickable": 0,
    #     "scrollable": 0,
    #     "checkable": 0,
    #     "long_clickable": 0,
    #     "editable": 0
    # }

    for window in forest.windows:
        if window.tree.nodes and 'com.google.android.inputmethod' in window.tree.nodes[0].package_name:
            continue  # we ignore input keyboards

        for node in window.tree.nodes:
            node_display_id = None
            if not node.child_ids or node.content_description or node.is_scrollable:

                if exclude_invisible_elements and not node.is_visible_to_user:
                    node_display_id = None
                elif node.package_name == "com.android.systemui" and (
                    (node.view_id_resource_name and "notificationIcons" in node.view_id_resource_name)
                    or
                    (node.content_description and "notification" in node.content_description.lower(
                    ))
                ):
                    node_display_id = None
                else:
                    node_display_id = display_id_counter

            extra_attributes[(window.id, node.unique_id)] = {
                "display_id": node_display_id}

            if node_display_id is not None:
                display_id_counter += 1

            # if safe_ele_get(node, 'is_clickable'):
            #     action_counts["clickable"] += 1
            # if safe_ele_get(node, 'is_scrollable'):
            #     action_counts["scrollable"] += 1
            # if safe_ele_get(node, 'is_checkable'):
            #     action_counts["checkable"] += 1
            # if safe_ele_get(node, 'is_long_clickable'):
            #     action_counts["long_clickable"] += 1
            # if safe_ele_get(node, 'is_editable'):
            #     action_counts["editable"] += 1

    def format_node(node, window_id, indent=0):
        """Recursively formats a node and its children as HTML."""
        indentation = "  " * (indent)
        child_indent = "  " * (indent + 1)
        elements = []

        # Get the HTML-like text description for this node
        element_id = extra_attributes.get(
            (window_id, node.unique_id), {}).get("display_id")
        node_html = node_to_text(node, element_id)

        # # Skip nodes that return an empty string from node_to_text
        # if not node_html:
        #     return ""

        # Add node's HTML content if it has the display id.
        if node_html != '' and element_id is not None:
            elements.append(f"{child_indent}{node_html}")

        child_elements = []
        # Recursively add child nodes if they exist and have valid HTML
        if node.child_ids:
            for child_id in node.child_ids:
                # Find the child node by ID within the same window
                child_node = next(
                    (n for n in window.tree.nodes if n.unique_id == child_id), None)
                if child_node and (window_id, child_id) not in processed_nodes:
                    processed_nodes.add((window_id, child_id))
                    child_html = format_node(child_node, window_id, indent + 1)
                    if child_html:  # Only add non-empty child nodes
                        if node.is_clickable:
                            if '<p' in child_html:
                                child_html = child_html.replace(
                                    "<p", "<button").replace("</p>", "</button>")
                        child_elements.append(child_html)

            if child_elements:
                if len(child_elements) == 1:
                    # If exactly one child block, append it directly (no extra <div>)
                    elements.append(child_elements[0])
                else:
                    children_html = f"{child_indent}<div>\n" + \
                        "\n".join(child_elements) + f"\n{child_indent}</div>"
                    elements.append(children_html)

        if len(elements) > 1:
            return f"{indentation}<div>\n" + "\n".join(elements) + f"\n{indentation}</div>"
        if len(elements) == 1:
            return elements[0]
        else:
            return ''

    # total_actions = sum(action_counts.values())
    # # Save action counts and total actions to count.txt
    # with open("count.txt", "a") as file:
    #     for action, count in action_counts.items():
    #         file.write(f"{action}: {count}\n")
    #     file.write(f"total_actions: {total_actions}\n\n")

    # Initialize the HTML output
    html_output = []

    for window in forest.windows:
        if window.tree.nodes and 'com.google.android.inputmethod' in window.tree.nodes[0].package_name:
            continue  # we ignore input keyboards
        processed_nodes = set()  # Track processed nodes to avoid duplication

        for node in window.tree.nodes:
            if (window.id, node.unique_id) not in processed_nodes:
                processed_nodes.add((window.id, node.unique_id))
                node_output = format_node(node, window.id, indent=1)
                html_output.append(node_output)

    # Wrap the entire structure in a root div or body tag
    html_desc = "<div>\n" + "\n".join(html_output) + "\n</div>"
    # html_desc = mask_irrelevant_info(html_desc)
    return html_desc


def aggregate_html_cleanup(html_str: str, indent_spaces=2) -> str:
    """
    Aggregates three post-processing steps:
    1) Removing empty <div> nodes.
    2) Merging text nodes into <input> elements.
    3) Producing one tag per line with a DFS pretty-print.
    4) Finalizing attribute formatting (e.g. numeric IDs and text attribute quotes).

    """
    # Parse the HTML once.
    soup = BeautifulSoup(html_str, "html.parser")

    # Remove empty <div> nodes by iterating over them.
    for div in soup.find_all("div"):
        if not div.get_text(strip=True) and not div.find(True):
            div.decompose()

    # Merge text into <input> elements.
    merge_text_into_input(soup)

    # Pretty-print the updated soup using DFS.
    lines = []
    for child in soup.contents:
        if isinstance(child, Tag):
            lines.extend(dfs_pretty_print_no_selfclose(
                child, level=0, indent_spaces=indent_spaces))
        elif isinstance(child, NavigableString):
            text = child.strip()
            if text:
                lines.append(text)
    cleaned_html = "\n".join(lines)

    # Finalize attribute formatting:
    # 1) Remove quotes from numeric IDs: id="123" => id=123.
    # 2) Convert text="..." to text='...'.
    cleaned_html = re.sub(r'id="(\d+)"', r'id=\1', cleaned_html)
    cleaned_html = re.sub(r'text="([^"]*)"', r"text='\1'", cleaned_html)

    # Remove any repeated blank lines.
    cleaned_html = re.sub(r'\n\s*\n', '\n', cleaned_html)

    return cleaned_html


def turn_tree_to_html_input_v2(
    forest: android_accessibility_forest_pb2.AndroidAccessibilityForest | Any,
    exclude_invisible_elements: bool = True,
    indent_spaces: int = 2
) -> str:
    """
    Traverses an accessibility tree and converts it into nested HTML, returning
    a string with clean and consistent indentation. This version aggregates the extra
    attribute calculation, node indexing, and DFS-based string creation.
    """
    display_id_counter = 0
    extra_attributes = {}
    html_lines = ["<div>"]

    for window in forest.windows:
        # Ignore input keyboards.
        if window.tree.nodes and 'com.google.android.inputmethod' in window.tree.nodes[0].package_name:
            continue

        # Build a dictionary for quick node lookup and compute extra attributes.
        node_dict = {}
        for node in window.tree.nodes:
            node_dict[node.unique_id] = node
            node_display_id = None
            if not node.child_ids or node.content_description or node.is_scrollable:
                if exclude_invisible_elements and not node.is_visible_to_user:
                    node_display_id = None
                elif node.package_name == "com.android.systemui" and (
                    (node.view_id_resource_name and "notificationIcons" in node.view_id_resource_name)
                    or (node.content_description and "notification" in node.content_description.lower())
                ):
                    node_display_id = None
                else:
                    node_display_id = display_id_counter
            extra_attributes[(window.id, node.unique_id)] = {
                "display_id": node_display_id}
            if node_display_id is not None:
                display_id_counter += 1

        processed_nodes = set()

        # Process each top-level node in this window.
        for node in window.tree.nodes:
            if (window.id, node.unique_id) not in processed_nodes:
                processed_nodes.add((window.id, node.unique_id))
                node_lines = format_node_clean_v2(
                    node, window.id, node_dict, extra_attributes, processed_nodes,
                    level=1, indent_spaces=indent_spaces
                )
                html_lines.extend(node_lines)

    html_lines.append("</div>")
    return "\n".join(html_lines)


def format_node_clean_v2(node, window_id, node_dict, extra_attributes, processed_nodes, level=1, indent_spaces=2) -> list[str]:
    """
    Recursively formats a node and its children into a list of strings,
    each representing one line of HTML with clean, consistent indentation.

    If there is exactly one child or if the children block is a single
    redundant <div> wrapper (see flatten_if_redundant), its indentation is decreased and its
    text is combined with the parent's line (if parent's output exists).

    Parameters:
      - node: current node to process.
      - window_id: id of the current window (for extra_attributes indexing).
      - node_dict: dict mapping node unique_id to node (for O(1) lookup).
      - extra_attributes: dict with precomputed extra attributes (like display_id).
      - processed_nodes: set of processed (window.id, node.unique_id) tuples.
      - level: current indentation level (starting at 1 for child nodes).
      - indent_spaces: number of spaces per indentation level.
    """
    indent = " " * (level * indent_spaces)
    lines = []

    element_id = extra_attributes.get(
        (window_id, node.unique_id), {}).get("display_id")
    node_html = node_to_text(node, element_id)
    if node_html and element_id is not None:
        # Place the node's HTML on one line at the proper indentation.
        lines.append(indent + node_html)

    # Process child nodes (if any).
    child_lines = []
    if node.child_ids:
        for child_id in node.child_ids:
            child_node = node_dict.get(child_id)
            if child_node and (window_id, child_id) not in processed_nodes:
                processed_nodes.add((window_id, child_id))
                child_lines.extend(
                    format_node_clean_v2(child_node, window_id, node_dict, extra_attributes, processed_nodes,
                                         level=level+1, indent_spaces=indent_spaces)
                )

        if child_lines:
            if len(child_lines) == 1:
                # There is exactly one child.
                expected_child_indent = " " * ((level+1) * indent_spaces)
                adjusted_child_line = child_lines[0]
                if adjusted_child_line.startswith(expected_child_indent):
                    adjusted_child_line = " " * \
                        (level * indent_spaces) + \
                        adjusted_child_line[len(expected_child_indent):]
                # If we have a parent's line, append the adjusted child line to it.
                if lines:
                    lines[-1] = lines[-1] + " " + adjusted_child_line.strip()
                else:
                    lines.append(" " * (level * indent_spaces) +
                                 adjusted_child_line.strip())
            else:
                # More than one child: try to flatten if possible.
                temp_lines = []
                temp_lines.append(indent + "<div>")
                temp_lines.extend(child_lines)
                temp_lines.append(indent + "</div>")
                child_block = "\n".join(temp_lines)
                flattened = flatten_if_redundant(
                    child_block, level, indent_spaces)
                if flattened is not None:
                    lines.append(indent + "<div>")
                    for f_line in flattened.splitlines():
                        lines.append(f_line)
                    lines.append(indent + "</div>")
                else:
                    lines.extend(temp_lines)

    return lines


def flatten_if_redundant(child_block: str, level: int, indent_spaces: int) -> str:
    pattern = r"^\s*<div>\s*\n\s*<div>\s*\n(.*)\n\s*</div>\s*\n\s*</div>\s*$"
    m = re.match(pattern, child_block, re.DOTALL)
    if m:
        inner = m.group(1)
        # Adjust indentation: remove one indent level (indent_spaces characters) from each line.
        adjusted_lines = []
        for line in inner.splitlines():
            if line.startswith(" " * indent_spaces):
                adjusted_lines.append(line[indent_spaces:])
            else:
                adjusted_lines.append(line)
        flattened_block = "\n".join(adjusted_lines)
        if flattened_block.strip():
            return flattened_block
    return None


def node_to_text(node, element_id, remove_time_and_ip=False):
    """Convert node attributes to a formatted HTML-compatible representation."""

    text_frame = "<p id=@ text='&'>#</p>"
    btn_frame = "<button id=@ text='&'>#</button>"
    checkbox_frame = "<checkbox id=@ checked=$ text='&'>#</checkbox>"
    input_frame = "<input id=@ text='&'>#</input>"

    clickable = safe_ele_get(node, 'is_clickable')
    scrollable = safe_ele_get(node, 'is_scrollable')
    checkable = safe_ele_get(node, 'is_checkable')
    long_clickable = safe_ele_get(node, 'is_long_clickable')

    editable = safe_ele_get(node, 'is_editable')
    actionable = clickable or scrollable or checkable or long_clickable or editable
    checked = safe_ele_get(node, 'is_checked', default=False)
    selected = safe_ele_get(node, 'is_selected', default=False)
    content_description = safe_ele_get(node, 'content_description', default='')
    view_text = safe_ele_get(node, 'text', default='')

    view_id_resource_name = safe_ele_get(
        node, 'view_id_resource_name', default='')
    class_name = safe_ele_get(node, 'class_name', default='')
    package_name = safe_ele_get(node, 'package_name', default='')
    # element_id = safe_ele_get(node, 'display_id', default='')
    # pdb.set_trace()

    if not content_description and not view_text and not actionable:
        return ''

    if package_name == "com.google.android.contacts" and class_name == "android.widget.Spinner":
        editable = False

    if editable:
        if view_text:
            view_desc = input_frame.replace(
                '@', str(element_id)).replace('#', view_text)
        else:
            view_desc = input_frame.replace(
                '@', str(element_id)).replace('#', '')
        if content_description:
            view_desc = view_desc.replace('&', content_description)
        else:
            view_desc = view_desc.replace(" text='&'", "")

    elif checkable:
        if view_text:
            view_desc = checkbox_frame.replace(
                '@', str(element_id)).replace('#', view_text)
        else:
            view_desc = checkbox_frame.replace(
                '@', str(element_id)).replace('#', '')
        if str(checked or selected):
            view_desc = view_desc.replace('$', str(checked or selected))
        else:
            view_desc = view_desc.replace(' checked=$', '')
        if content_description:
            view_desc = view_desc.replace('&', content_description)
        else:
            view_desc = view_desc.replace(" text='&'", "")

    elif clickable or (class_name == "android.widget.RadialTimePickerView$RadialPickerTouchHelper" and content_description):  # or long_clickable
        btn_frame = "<button id=@ text='&'>#</button>"

        if element_id == '':
            return ''

        view_desc = btn_frame.replace('@', str(element_id))
        view_desc = view_desc.replace('#', view_text)

        if content_description:
            view_desc = view_desc.replace('&', content_description)
        elif view_text == '':
            resource_desc = safe_ele_get(node, 'resource_name', default='')
            resource_desc = resource_desc.split('/')[-1]

            view_id_resource_name = safe_ele_get(
                node, 'view_id_resource_name', default='')
            view_id_resource_name = view_id_resource_name.split('/')[-1]

            if resource_desc == 'sd_main_fab':
                resource_desc = 'Create one event or task'
            if view_id_resource_name == 'sd_main_fab':
                resource_desc = 'Create one event or task'

            if resource_desc != '':
                view_desc = view_desc.replace('&', resource_desc)
            elif view_id_resource_name != '':
                view_desc = view_desc.replace('&', view_id_resource_name)
            else:
                view_desc = view_desc.replace(" text='&'", "")
        else:
            view_desc = view_desc.replace(" text='&'", "")
    elif scrollable:
        return ''
    else:
        if view_text:
            view_desc = text_frame.replace(
                '@', str(element_id)).replace('#', view_text)
        else:
            view_desc = text_frame.replace(
                '@', str(element_id)).replace('#', '')

        if content_description:
            view_desc = view_desc.replace('&', content_description)
        else:
            view_desc = view_desc.replace(" text='&'", "")

        if ((view_id_resource_name and ('slider' in view_id_resource_name.lower()))
                or (content_description and content_description.lower().strip() in volume_keywords)):
            view_desc = view_desc.replace('<p', '<button').replace(
                "</p>", "</button>")  # we still take them as a button
    return view_desc


def safe_ele_get(view_obj, key, default=None):
    value = getattr(view_obj, key, default)
    return value if value is not None else default


def determine_html_tag(node):
    """Determine the appropriate HTML tag based on the node class or description."""
    # Mapping of node class names or types to HTML tags
    if "button" in node.class_name.lower():
        return "button"
    elif "input" in node.class_name.lower():
        return "input"
    elif "checkbox" in node.class_name.lower():
        return "input type='checkbox'"
    elif "text" in node.class_name.lower():
        return "p"
    else:
        return "div"  # Default to a <div> for generic containers


def extract_actions_with_display_id(
    forest: android_accessibility_forest_pb2.AndroidAccessibilityForest | Any,
    exclude_invisible_elements: bool = True,
) -> dict:
    """
    Extracts all available actions from the accessibility forest.
    Returns a dictionary mapping display IDs to their available actions.
    """

    # actions_map = {}  # Dictionary to store actions for each display ID
    display_id_counter = 0

    action_templates = {
        "clickable": '{{"action_type": "click", "index": {index}}}',
        "long_clickable": '{{"action_type": "long_press", "index": {index}}}',
        "editable": '{{"action_type": "input_text", "text": "<text_input>", "index": {index}}}',
        "scrollable": '{{"action_type": "scroll", "direction": "<direction>", "index": {index}}}',
        "checkable": '{{"action_type": "click", "index": {index}}}',
    }
    action_list = []

    for window in forest.windows:
        for node in window.tree.nodes:
            node_display_id = None
            if not node.child_ids or node.content_description or node.is_scrollable:
                if exclude_invisible_elements and not node.is_visible_to_user:
                    node_display_id = None
                else:
                    node_display_id = display_id_counter

            if node_display_id is not None:
                display_id_counter += 1

                formatted_actions = []

                if safe_ele_get(node, 'is_clickable') or safe_ele_get(node, 'is_checkable'):
                    formatted_actions.append(
                        action_templates["clickable"].format(
                            index=node_display_id)
                    )
                if safe_ele_get(node, 'is_long_clickable'):
                    formatted_actions.append(
                        action_templates["long_clickable"].format(
                            index=node_display_id)
                    )
                if safe_ele_get(node, 'is_editable'):
                    formatted_actions.append(
                        action_templates["editable"].format(
                            index=node_display_id)
                    )
                if safe_ele_get(node, 'is_scrollable'):
                    formatted_actions.append(
                        action_templates["scrollable"].format(
                            index=node_display_id)
                    )

                if formatted_actions:
                    # actions_map[node_display_id] = formatted_actions
                    action_list.extend(formatted_actions)

    default_actions = [
        '{"action_type": "keyboard_enter"}',
        '{"action_type": "navigate_home"}',
        '{"action_type": "navigate_back"}',
        '{"action_type": "open_app", "app_name": "<name>"}',
        '{"action_type": "wait"}',
        '{"action_type": "status", "goal_status": "complete"}',
        '{"action_type": "answer", "text": "<answer_text>"}',
    ]

    action_list.extend(default_actions)
    return action_list


def extract_actions_with_display_id_v2(
    forest: android_accessibility_forest_pb2.AndroidAccessibilityForest | Any,
    exclude_invisible_elements: bool = True,
    return_mapping: bool = False,
    refine_a11y_tree: bool = False,
    family: str = 'android_world',
) -> list:
    """
    Extracts all available actions from the accessibility forest and modifies the action list
    based on parent-child relationships.
    Returns a list of available actions.
    """

    display_id_counter = 0

    # Define action templates
    action_templates = {
        "clickable": '{{"action_type": "click", "index": {index}}}',
        "long_clickable": '{{"action_type": "long_press", "index": {index}}}',
        "editable": '{{"action_type": "input_text", "text": "<text_input>", "index": {index}}}',
        "clearable": '{{"action_type": "clear_text", "index": {index}}}',
        "scrollable": '{{"action_type": "scroll", "direction": "{direction}", "index": {index}}}',
        "checkable": '{{"action_type": "click", "index": {index}}}',
        "scrollbar": '{{"action_type": "scroll", "direction": "{direction}"}}',
    }
    action_list = []

    extra_attributes = {}

    # Recursive helper function to extract actions

    for window in forest.windows:
        if 'com.google.android.inputmethod' in window.tree.nodes[0].package_name:
            continue  # we ignore input keyboards
        for node in window.tree.nodes:
            node_display_id = None
            if not node.child_ids or node.content_description or node.is_scrollable:
                if exclude_invisible_elements and not node.is_visible_to_user:
                    node_display_id = None
                elif node.package_name == "com.android.systemui" and (
                    (node.view_id_resource_name and "notificationIcons" in node.view_id_resource_name)
                    or
                    (node.content_description and "notification" in node.content_description.lower(
                    ))
                ):
                    node_display_id = None
                else:
                    node_display_id = display_id_counter

            if node.HasField("bounds_in_screen"):
                left = node.bounds_in_screen.left
                top = node.bounds_in_screen.top
                right = node.bounds_in_screen.right
                bottom = node.bounds_in_screen.bottom
            else:
                left = top = right = bottom = 0

            extra_attributes[(window.id, node.unique_id)] = {"display_id": node_display_id,
                                                             "bounds": (left, top, right, bottom), }

            if node_display_id is not None:
                display_id_counter += 1

    def process_node(node, window_id, parent_node=None):

        node_display_id = extra_attributes.get(
            (window_id, node.unique_id), {}).get("display_id")

        formatted_actions = []

        # Determine actions for the current node
        if node_display_id is not None:

            class_name = safe_ele_get(node, 'class_name')
            view_id_resource_name = safe_ele_get(
                node, 'view_id_resource_name', default='')
            content_description = safe_ele_get(
                node, 'content_description', default='')
            editable = safe_ele_get(node, 'is_editable') and not (
                safe_ele_get(
                    node, 'package_name') == "com.google.android.contacts"
                and class_name == "android.widget.Spinner"
            )

            if safe_ele_get(node, 'is_clickable') or safe_ele_get(node, 'is_checkable'):
                formatted_actions.append(
                    action_templates["clickable"].format(index=node_display_id)
                )
                formatted_actions.append(
                    action_templates["long_clickable"].format(
                        index=node_display_id)
                )
                node_text = (node.text or "") + " " + \
                    (node.content_description or "")
                if refine_a11y_tree and "search mode" in node_text.lower():
                    formatted_actions.append(
                        action_templates["editable"].format(
                            index=node_display_id)
                    )
            elif parent_node and parent_node.is_clickable:  # Modify actions for child nodes based on parent attributes
                # Add a custom action if the parent is clickable
                formatted_actions.append(
                    action_templates["clickable"].format(index=node_display_id)
                )

                formatted_actions.append(
                    action_templates["long_clickable"].format(
                        index=node_display_id)
                )
                node_text = (node.text or "") + " " + \
                    (node.content_description or "")
                if refine_a11y_tree and "search mode" in node_text.lower():
                    formatted_actions.append(
                        action_templates["editable"].format(
                            index=node_display_id)
                    )

            if class_name == "android.widget.RadialTimePickerView$RadialPickerTouchHelper":
                node_text_stripped = (node.content_description or "").strip()
                if node_text_stripped and action_templates["clickable"].format(index=node_display_id) not in formatted_actions:
                    formatted_actions.append(
                        action_templates["clickable"].format(
                            index=node_display_id)
                    )

            if editable:
                formatted_actions.append(
                    action_templates["editable"].format(index=node_display_id)
                )
                formatted_actions.append(
                    action_templates["clearable"].format(
                        index=node_display_id),
                    # we add one additional action  ## not yet for scroll version.
                )

            if safe_ele_get(node, 'is_scrollable'):
                formatted_actions.append(
                    action_templates["scrollable"].format(
                        index=node_display_id, direction="up")
                )
                formatted_actions.append(
                    action_templates["scrollable"].format(
                        index=node_display_id, direction="down")
                )
                formatted_actions.append(
                    action_templates["scrollable"].format(
                        index=node_display_id, direction="left")
                )
                formatted_actions.append(
                    action_templates["scrollable"].format(
                        index=node_display_id, direction="right")
                )

            if not safe_ele_get(node, 'is_scrollable'):

                if view_id_resource_name and ('slider' in view_id_resource_name.lower()):
                    formatted_actions.append(
                        action_templates["scrollable"].format(
                            index=node_display_id, direction="left")
                    )
                    formatted_actions.append(
                        action_templates["scrollable"].format(
                            index=node_display_id, direction="right")
                    )

                elif (content_description and content_description.lower().strip() in volume_keywords):
                    formatted_actions.append(
                        action_templates["scrollable"].format(
                            index=node_display_id, direction="left")
                    )
                    formatted_actions.append(
                        action_templates["scrollable"].format(
                            index=node_display_id, direction="right")
                    )

        # Add actions to the main action list
        if formatted_actions:
            action_list.extend(formatted_actions)

        # Process child nodes recursively
        for child_id in node.child_ids:
            child_node = next(
                (n for n in window.tree.nodes if n.unique_id == child_id), None)
            if child_node and (window_id, child_id) not in processed_nodes:
                processed_nodes.add((window.id, child_id))
                process_node(child_node, window_id, parent_node=node)

    # Process the entire forest
    for window in forest.windows:
        if 'com.google.android.inputmethod' in window.tree.nodes[0].package_name:
            continue  # we ignore input keyboards
        processed_nodes = set()  # Track processed nodes to avoid duplication
        for node in window.tree.nodes:
            if (window.id, node.unique_id) not in processed_nodes:
                processed_nodes.add((window.id, node.unique_id))
                process_node(node, window.id)

    # Add default actions

    default_scroll = ['{"action_type": "scroll", "direction": "up"}',
                      '{"action_type": "scroll", "direction": "down"}',
                      '{"action_type": "scroll", "direction": "left"}',
                      '{"action_type": "scroll", "direction": "right"}',]

    default_actions = [
        '{"action_type": "navigate_home"}',
        '{"action_type": "navigate_back"}',
        '{"action_type": "open_app", "app_name": "<name>"}',
        '{"action_type": "wait"}',
        '{"action_type": "status", "goal_status": "complete"}',
        '{"action_type": "answer", "text": "<answer_text>"}',
    ]

    if family == "android_control":
        action_list.extend(default_scroll)

    action_list.extend(default_actions)

    if return_mapping:
        return action_list, extra_attributes
    else:
        return action_list


def extract_actions_with_display_id_v3(
    forest: android_accessibility_forest_pb2.AndroidAccessibilityForest | Any,
    exclude_invisible_elements: bool = True,
    return_mapping: bool = False,
    refine_a11y_tree: bool = False,
    family: str = 'android_world',
) -> tuple:
    """
    Extracts all available actions from the accessibility forest and modifies the action list
    based on parent-child relationships. Also counts interactable UI elements.
    Returns a list of available actions and the count of interactable elements.
    """

    display_id_counter = 0
    interactable_count = 0  # Counter for interactable UI elements

    # Define action templates
    action_templates = {
        "clickable": '{{"action_type": "click", "index": {index}}}',
        "long_clickable": '{{"action_type": "long_press", "index": {index}}}',
        "editable": '{{"action_type": "input_text", "text": "<text_input>", "index": {index}}}',
        "clearable": '{{"action_type": "clear_text", "index": {index}}}',
        "scrollable": '{{"action_type": "scroll", "direction": "{direction}", "index": {index}}}',
        "checkable": '{{"action_type": "click", "index": {index}}}',
        "scrollbar": '{{"action_type": "scroll", "direction": "{direction}"}}',
    }
    action_list = []
    extra_attributes = {}

    for window in forest.windows:
        if 'com.google.android.inputmethod' in window.tree.nodes[0].package_name:
            continue  # Ignore input keyboards

        for node in window.tree.nodes:
            node_display_id = None
            if not node.child_ids or node.content_description or node.is_scrollable:
                if exclude_invisible_elements and not node.is_visible_to_user:
                    node_display_id = None
                elif node.package_name == "com.android.systemui" and (
                    (node.view_id_resource_name and "notificationIcons" in node.view_id_resource_name)
                    or
                    (node.content_description and "notification" in node.content_description.lower(
                    ))
                ):
                    node_display_id = None
                else:
                    node_display_id = display_id_counter

            if node.HasField("bounds_in_screen"):
                left, top = node.bounds_in_screen.left, node.bounds_in_screen.top
                right, bottom = node.bounds_in_screen.right, node.bounds_in_screen.bottom
            else:
                left = top = right = bottom = 0

            extra_attributes[(window.id, node.unique_id)] = {
                "display_id": node_display_id,
                "bounds": (left, top, right, bottom),
            }

            if node_display_id is not None:
                display_id_counter += 1

    def process_node(node, window_id, parent_node=None):
        nonlocal interactable_count  # Use the interactable counter

        node_display_id = extra_attributes.get(
            (window_id, node.unique_id), {}).get("display_id")

        formatted_actions = []

        # Determine actions for the current node
        if node_display_id is not None:
            if safe_ele_get(node, 'is_clickable') or safe_ele_get(node, 'is_checkable'):
                formatted_actions.append(
                    action_templates["clickable"].format(index=node_display_id)
                )
                formatted_actions.append(
                    action_templates["long_clickable"].format(
                        index=node_display_id)
                )
                node_text = (node.text or "") + " " + \
                    (node.content_description or "")
                if refine_a11y_tree and "search mode" in node_text.lower():
                    formatted_actions.append(
                        action_templates["editable"].format(
                            index=node_display_id)
                    )
            elif parent_node and parent_node.is_clickable:  # Modify actions for child nodes based on parent attributes
                # Add a custom action if the parent is clickable
                formatted_actions.append(
                    action_templates["clickable"].format(index=node_display_id)
                )
                formatted_actions.append(
                    action_templates["long_clickable"].format(
                        index=node_display_id)
                )
                node_text = (node.text or "") + " " + \
                    (node.content_description or "")
                if refine_a11y_tree and "search mode" in node_text.lower():
                    formatted_actions.append(
                        action_templates["editable"].format(
                            index=node_display_id)
                    )

            if safe_ele_get(node, 'class_name') == "android.widget.RadialTimePickerView$RadialPickerTouchHelper":
                node_text_stripped = (node.content_description or "").strip()
                if node_text_stripped and action_templates["clickable"].format(index=node_display_id) not in formatted_actions:
                    formatted_actions.append(
                        action_templates["clickable"].format(
                            index=node_display_id)
                    )

            if safe_ele_get(node, 'is_editable'):
                formatted_actions.append(
                    action_templates["editable"].format(index=node_display_id)
                )
                formatted_actions.append(
                    action_templates["clearable"].format(
                        index=node_display_id),
                )

            view_id_resource_name = safe_ele_get(
                node, 'view_id_resource_name', default='')
            content_description = safe_ele_get(
                node, 'content_description', default='')
            if safe_ele_get(node, 'is_scrollable'):
                formatted_actions.extend([
                    action_templates["scrollable"].format(
                        index=node_display_id, direction=direction)
                    for direction in ["up", "down", "left", "right"]
                ])

            if not safe_ele_get(node, 'is_scrollable'):
                if view_id_resource_name and ('slider' in view_id_resource_name.lower()):
                    formatted_actions.extend([
                        action_templates["scrollable"].format(
                            index=node_display_id, direction="left"),
                        action_templates["scrollable"].format(
                            index=node_display_id, direction="right"),
                    ])
                elif content_description and content_description.lower().strip() in volume_keywords:
                    formatted_actions.extend([
                        action_templates["scrollable"].format(
                            index=node_display_id, direction="left"),
                        action_templates["scrollable"].format(
                            index=node_display_id, direction="right"),
                    ])

        # If any actions are assigned to this node, increase interactable count
        if formatted_actions:
            interactable_count += 1
            action_list.extend(formatted_actions)

        # Process child nodes recursively
        for child_id in node.child_ids:
            child_node = next(
                (n for n in window.tree.nodes if n.unique_id == child_id), None)
            if child_node and (window_id, child_id) not in processed_nodes:
                processed_nodes.add((window.id, child_id))
                process_node(child_node, window_id, parent_node=node)

    # Process the entire forest
    for window in forest.windows:
        if 'com.google.android.inputmethod' in window.tree.nodes[0].package_name:
            continue  # Ignore input keyboards
        processed_nodes = set()  # Track processed nodes to avoid duplication
        for node in window.tree.nodes:
            if (window.id, node.unique_id) not in processed_nodes:
                processed_nodes.add((window.id, node.unique_id))
                process_node(node, window.id)

    # Add default actions
    default_scroll = ['{"action_type": "scroll", "direction": "up"}',
                      '{"action_type": "scroll", "direction": "down"}',
                      '{"action_type": "scroll", "direction": "left"}',
                      '{"action_type": "scroll", "direction": "right"}']

    default_actions = [
        '{"action_type": "navigate_home"}',
        '{"action_type": "navigate_back"}',
        '{"action_type": "open_app", "app_name": "<name>"}',
        '{"action_type": "wait"}',
        '{"action_type": "status", "goal_status": "complete"}',
        '{"action_type": "answer", "text": "<answer_text>"}',
    ]

    if family == "android_control":
        action_list.extend(default_scroll)

    action_list.extend(default_actions)

    if return_mapping:
        return action_list, interactable_count, extra_attributes
    else:
        return action_list, interactable_count


def html_truncate(tokenizer, html_desc: str, max_tokens=2800):
    tokens = tokenizer.tokenize(html_desc)

    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        truncated_html_desc = tokenizer.convert_tokens_to_string(
            truncated_tokens)
    else:
        truncated_html_desc = html_desc

    return truncated_html_desc

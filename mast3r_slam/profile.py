import time
from functools import wraps
from collections import defaultdict
from contextlib import contextmanager
import inspect

# Modified registry initialization
def create_timing_dict():
    return {
        'total_time': 0,
        'call_count': 0,
        'children': defaultdict(create_timing_dict)
    }

timing_registry = defaultdict(create_timing_dict)
_current_stack = []

@contextmanager
def timeblock(block_name):
    global timing_registry, _current_stack
    
    parent = _current_stack[-1] if _current_stack else None
    _current_stack.append(block_name)
    
    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Update timing for current block
    timing_registry[block_name]['total_time'] += execution_time
    timing_registry[block_name]['call_count'] += 1
    
    # Update parent's children if there is a parent
    if parent:
        timing_registry[parent]['children'][block_name]['total_time'] += execution_time
        timing_registry[parent]['children'][block_name]['call_count'] += 1
    
    _current_stack.pop()

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global timing_registry, _current_stack
        
        func_name = func.__name__
        parent = _current_stack[-1] if _current_stack else None
        _current_stack.append(func_name)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Update timing for current function
        timing_registry[func_name]['total_time'] += execution_time
        timing_registry[func_name]['call_count'] += 1
        
        # Update parent's children if there is a parent
        if parent:
            timing_registry[parent]['children'][func_name]['total_time'] += execution_time
            timing_registry[parent]['children'][func_name]['call_count'] += 1
        
        _current_stack.pop()
        return result
    return wrapper

def print_timing_registry(indent="   ", root=None, output_file=None):
    # Create output lines
    output = ["\nTiming Statistics:",
              "Function Name\tTotal Time (s)\tCalls\tAvg Time (s)",
              "-" * 70]
    
    def print_entry(name, info, depth=0):
        total_time = info['total_time']
        calls = info['call_count']
        avg_time = total_time / calls if calls > 0 else 0
        
        indented_name = indent * depth + name
        output.append(f"{indented_name:<40}\t{total_time:.4f}\t{calls}\t{avg_time:.4f}")
        
        # Get children from the registry if this is a top-level function
        children_info = info['children']
        if not children_info and name in timing_registry:
            children_info = timing_registry[name]['children']
        
        sorted_children = sorted(
            children_info.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        
        for child_name, child_info in sorted_children:
            print_entry(child_name, child_info, depth + 1)
    
    if root is None:
        top_level_funcs = []
        for func_name, info in timing_registry.items():
            is_child = False
            for parent_info in timing_registry.values():
                if func_name in parent_info['children']:
                    is_child = True
                    break
            if not is_child:
                top_level_funcs.append((func_name, info))
        
        sorted_funcs = sorted(
            top_level_funcs,
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        for func_name, info in sorted_funcs:
            print_entry(func_name, info)
    else:
        print_entry(root, timing_registry[root])
    
    # Join all lines
    full_output = '\n'.join(output)
    
    # Either print or write to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_output)
    else:
        print(full_output)

def reset_timing_registry():
    global timing_registry, _current_stack
    timing_registry = defaultdict(create_timing_dict)
    _current_stack = []
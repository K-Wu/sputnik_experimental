def emit_launch_config():
    """for now, emit 1) grid+block configuration in comments or a dummy main() function, 2) launch bound in the function signature of the original function or a dedicated bounded wrapper function"""
    pass


def emit_constants():
    """There are several ways for constants to be passed into kernel 1) constants are defined as device constants literals in the source file and passed as parameters
    2) constants are defined as device constants literals in the source file and referred to directly 3) constants are in normal allocated cuda device memory and passed as parameters
    We need to emit constants in the case of 1) and 2)"""
    pass

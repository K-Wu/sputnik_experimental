from collections import namedtuple

# TODO: establish a clear hierarchy like Kernel boundary -> parallel region -> statement

LaunchBounds = namedtuple("LaunchBounds", ["kernel_max_threads", "kernel_min_blocks"])
LaunchConfigGrid = namedtuple("LaunchConfigGrid", ["grid_x", "grid_y", "grid_z"])
LaunchConfigBlock = namedtuple("LaunchConfigBlock", ["block_x", "block_y", "block_z"])
LaunchConfig = namedtuple("LaunchConfig", ["grid", "block", "shared_memory", "stream"])


class TraversalTemplate:
    def __init__(self):
        self.type = None  # whether it is spmm, sddmm, edge-parallel or node-parallel
        self.adj_matrix_format = None
        self.template_str = None


class ParallelRegionSpecification:
    def __init__(self):
        self.name = None
        self.type = None  # whether it is spmm, sddmm, edge-parallel or node-parallel
        self.constant_literals = dict()  # constant literal defined in file scope
        self.constant_data = dict()  # constant data stored in regular device memory
        self.for_loop_schedule = None  # 1) assignment from semantic for-loop level to architecture level, 2) loop transformation sequence
        self.traversal_template = None
        self.macro_definitions = dict()
        self.launch_config_requirements = None
        self.statements = []


class KernelSpecification:
    def __init__(self):
        self.name = None  # kernel name
        self.launch_bound = None  # {}
        self.launch_config = None
        self.constant_literals = dict()  # constant literal defined in file scope
        self.constant_data = dict()  # constant data stored in regular device memory
        self.macro_definitions = dict()
        self.parallel_regions = []
        # NB: warp specialization is expressed by specifying a mega-kernel composed of several subkernels of the same launch grid configurations
        self.warp_specialized_kernels = None


class ProgramExecutionSpecification:
    def __init__(self):
        # TODO: we also need to specify hyperparameter, kernel-twisting macros, and perhaps task, i.e., node or edge classification, etc., in this class instance
        # specify dataset and load balance specifications
        raise NotImplementedError
        self.dataset_name = None
        self.data_load_cpp_func_emitter = None
        self.load_balance_specification = None

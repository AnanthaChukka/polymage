from __init__ import *

import sys
sys.path.insert(0, ROOT+"/apps/python/")

from cpp_compiler import *
from polymage_jacobi import stencil_jacobi
from exec_pipe import custom_exec_jacobi
from constructs import *

from compiler import *
import tuner

def auto_tune(app_data):
    pipe_data = app_data['pipe_data']
    app_name = app_data['app']

    stencil = stencil_jacobi(app_data)

    live_outs = [stencil]
    N = pipe_data['N']
    param_estimates = [(N, app_data['N'])]
    param_constraints = [ Condition(N, '==', app_data['N']) ]
    dst_path = "/tmp"

    group_size_configs = [2, 3, 4]

    tile_size_configs = []

    tile_size_configs.append([8, 8, 32])
    tile_size_configs.append([8, 8, 64])
    tile_size_configs.append([8, 8, 128])
    tile_size_configs.append([8, 16, 16])
    tile_size_configs.append([8, 16, 32])
    tile_size_configs.append([8, 16, 64])
    tile_size_configs.append([8, 32, 32])
    tile_size_configs.append([8, 32, 64])
    tile_size_configs.append([8, 64, 64])
    tile_size_configs.append([16, 16, 16])
    tile_size_configs.append([16, 16, 32])
    tile_size_configs.append([16, 16, 64])
    tile_size_configs.append([16, 32, 32])
    tile_size_configs.append([16, 32, 64])
    tile_size_configs.append([32, 32, 32])

    opts = []
    # relative path to root directory from app dir
    ROOT = app_data['ROOT']
    opts = []
    if app_data['early_free']:
        opts += ['early_free']
    if app_data['optimize_storage']:
        opts += ['optimize_storage']
    if app_data['pool_alloc']:
        opts += ['pool_alloc']
    if app_data['multipar']:
        opts += ['multipar']

    gen_compile_string(app_data)
    cxx_string = app_data['cxx_string']

    # Generate Variants for Tuning
    # ============================

    gen_config = {"_tuner_app_name": app_name,
                  "_tuner_live_outs": live_outs,
                  "_tuner_param_constraints": param_constraints, #optional
                  "_tuner_param_estimates": param_estimates, #optional
                  "_tuner_tile_size_configs": tile_size_configs, #optional
                  "_tuner_group_size_configs": group_size_configs, #optional
                  "_tuner_opts": opts, #optional
                  "_tuner_dst_path" : dst_path, # optional
                  "_tuner_cxx_string" : cxx_string, # optional
                  "_tuner_root_path" : ROOT, # needed if pool_alloc is set
                  "_tuner_debug_flag": True, # optional
                  "_tuner_opt_datadict": app_data
                 }

    _tuner_src_path, _tuner_configs_count, _tuner_pipe = \
        tuner.generate(gen_config)

    # Execute the generated variants
    # ==============================

    exec_config = {"_tuner_app_name": app_name,
                   "_tuner_pipe": _tuner_pipe,
                   "_tuner_src_path": _tuner_src_path, # optional
                   "_tuner_configs_count": _tuner_configs_count, # optional
                   "_tuner_omp_threads": 4, # optional
                   "_tuner_nruns": 1, # optional
                   "_tuner_debug_flag": True, # optional
                   "_tuner_custom_executor": custom_exec_jacobi,
                   "_tuner_app_data": app_data
                  }

    tuner.execute(exec_config)

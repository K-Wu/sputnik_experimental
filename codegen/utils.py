#!/usr/bin/env python3
import subprocess
import os


def check_git_existence():
    """Check if git is installed and available in the path."""
    try:
        subprocess.check_output(["git", "--version"])
    except OSError:
        raise OSError("Git is not installed. Please install git and try again.")


def get_git_root_path():
    """Get the root path of the git repository."""
    check_git_existence()
    return os.path.normpath(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )


def get_sputnik_root_path():
    """Get the root path of the sputnik repository."""
    current_project_root_path = get_git_root_path()
    # current_project_root_path could be either HET or sputnik
    if current_project_root_path.endswith("HET"):
        return os.path.join(current_project_root_path, "third_party", "sputnik")
    elif current_project_root_path.endswith("sputnik"):
        return current_project_root_path
    else:
        raise RuntimeError("Cannot find the root path of sputnik.")


def get_dir_for_op(op_name):
    """Get the directory for the operator."""
    return os.path.join(
        get_sputnik_root_path(), "sputnik", "generated", "gen_" + op_name
    )

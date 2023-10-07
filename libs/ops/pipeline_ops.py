# from blueprints import Blueprints
from copy import deepcopy
from pprint import pprint


def process_blueprint_groups():
    """
    combine blueprints in groups to single blueprint for loading purposes
    Returns:
    """
    module_groups = {}
    for group_name, blueprint_names in Blueprints.GROUPS.items():
        module_groups[group_name] = {}
        for blueprint_name in blueprint_names:
            for blueprint_info in Blueprints.BLUPRINTS[blueprint_name]:
                module_name, flags = blueprint_info[0], blueprint_info[1]
                if module_name not in module_groups[group_name]:
                    module_groups[group_name][module_name] = set([])
                module_groups[group_name][module_name].update(set(flags))

        module_groups[group_name] = [(module_name, list(module_groups[group_name][module_name])) for module_name in
                                     module_groups[group_name]]
    module_groups = {**module_groups, **deepcopy(Blueprints.BLUPRINTS)}
    return module_groups


# pprint(process_blueprint_groups())

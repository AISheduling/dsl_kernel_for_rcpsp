# Global Project Structure

The root object contains general metadata about the scheduling problem and stores collections of resources and tasks.

| Field            | Type     | Required | Description                                                                                                              |
| ---------------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| `schema_version` | `String` | Yes      | Version of the DSL schema (current version: `"0.1"`). Enables backward compatibility management as the language evolves. |
| `problem_id`     | `String` | Yes      | Unique identifier of the scheduling problem (e.g. `"T01_CLASSIC_SCALED"`).                                               |
| `domain`         | `String` | Yes      | Target application domain. Supported values: `"rcpsp"`, `"cluster"`.                                                     |
| `description`    | `String` | No       | Human-readable description of the scenario.                                                                              |
| `project`        | `Object` | Yes      | Object containing global project objectives (see structure below).                                                       |
| `extensions`     | `Object` | No       | Container for domain-specific global parameters (e.g. `locations`, `objectives`).                                        |
| `resources`      | `Array`  | Yes      | Collection of available resources.                                                                                       |
| `tasks`          | `Array`  | Yes      | Collection of project activities.                                                                                        |

## `project` Structure

| Field       | Type     | Description                                                                                                                                                               |
| ----------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`      | `String` | Project name.                                                                                                                                                             |
| `objective` | `String` | Primary optimization objective (e.g. `"minimize_makespan"`, `"minimize_completion_time"`). If multiple objectives are specified, the value should be `"multi_objective"`. |

# Resources

Resources represent any entities that constrain project execution, including personnel, budget, physical space, or computational infrastructure.

## Resource Core Schema

| Field        | Type     | Required | Description                                                        |
| ------------ | -------- | -------- | ------------------------------------------------------------------ |
| `id`         | `String` | Yes      | Unique resource identifier (e.g. `"R_WORKER"` or `"NODE_TYPE_A"`). |
| `name`       | `String` | No       | Human-readable resource name.                                      |
| `capacity`   | `Number` | Yes      | Total available quantity or capacity of the resource.              |
| `extensions` | `Object` | No       | Domain-specific resource attributes.                               |

## `extensions.rcpsp`

| Field                | Type      | Description                                                                                           |
| -------------------- | --------- | ----------------------------------------------------------------------------------------------------- |
| `type`               | `String`  | Resource type: `"renewable"` (e.g. workers, equipment) or `"non_renewable"` (e.g. materials, budget). |
| `cost_per_period`    | `Number`  | Cost of using the resource per unit of time.                                                          |
| `cost_per_unit`      | `Number`  | Cost of consuming one unit of the resource.                                                           |
| `is_zone`            | `Boolean` | Indicates that the resource represents a physical workspace or area.                                  |
| `is_global_capacity` | `Boolean` | Indicates that the capacity constraint applies globally across the entire site.                       |
| `location_dependent` | `Boolean` | Indicates that the resource is associated with a location and requires transfer time between sites.   |
| `initial_location`   | `String`  | Initial location of the resource.                                                                     |
| `transfer_times`     | `Object`  | Travel-time matrix in the form `{"From": {"To": Time}}`.                                              |

## `extensions.cluster`

In the cluster domain, resources typically represent compute nodes. Their `capacity` specifies the number of identical nodes available in the resource pool.

| Field           | Type     | Description                                     |
| --------------- | -------- | ----------------------------------------------- |
| `gpu_type`      | `String` | GPU model (e.g. `"A100"`, `"V100"`).            |
| `gpu_memory_gb` | `Number` | GPU memory available on a single node (GB).     |
| `ram_gb`        | `Number` | System memory available on a single node (GB).  |
| `cpu_cores`     | `Number` | Number of CPU cores available on a single node. |

# Tasks

Tasks represent nodes in the scheduling graph. Execution details, durations, and resource requirements are defined by the selected domain extension.

## Task Core Schema

| Field          | Type     | Required | Description                                                                        |
| -------------- | -------- | -------- | ---------------------------------------------------------------------------------- |
| `id`           | `String` | Yes      | Unique task identifier.                                                            |
| `name`         | `String` | No       | Human-readable task name.                                                          |
| `dependencies` | `Array`  | Yes      | Collection of predecessor relationships.                                           |
| `extensions`   | `Object` | Yes      | Domain-specific task attributes (required for storing durations and requirements). |

## `extensions.rcpsp`

| Field      | Type     | Description                                                                                                                                      |
| ---------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `location` | `String` | Location where the activity is executed.                                                                                                         |
| `modes`    | `Array`  | Alternative execution modes. Each mode contains `mode_id`, `duration`, and a collection of resource `requirements` (`resource_id` + `quantity`). |

## `extensions.cluster`

| Field            | Type      | Description                                                                                        |
| ---------------- | --------- | -------------------------------------------------------------------------------------------------- |
| `duration`       | `Number`  | Estimated execution time of the computational task.                                                |
| `is_preemptible` | `Boolean` | Indicates whether the scheduler may interrupt the task to free resources for higher-priority jobs. |
| `requirements`   | `Array`   | Hardware requirements for the task (see structure below).                                          |

## `requirements` Structure within `extensions.cluster`

Each requirement object specifies the amount of resources that must be allocated from a particular resource pool (`resource_id`).

* `resource_id` (`String`) — Identifier of the required resource pool.
* `gpu_count` (`Number`) — Number of GPUs required.
* `cpu_cores` (`Number`) — Number of CPU cores required.
* `ram_gb` (`Number`) — Amount of RAM required (GB).

# Dependencies

Dependencies define precedence relations between tasks and determine the topological structure of the schedule. They are stored in `task.dependencies`.

| Field        | Type     | Required | Description                                                                                                                                     |
| ------------ | -------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `task_id`    | `String` | Yes      | Identifier of the predecessor task.                                                                                                             |
| `type`       | `String` | Yes      | Dependency type. Supported values are `"FS"` (Finish-to-Start) and `"SS"` (Start-to-Start).                                                     |
| `extensions` | `Object` | No       | Additional dependency parameters. In RCPSP, this may include a `lag` field representing the delay after completion of the predecessor activity. |

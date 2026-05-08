# rendezvous_manager

This package isolates all new rendezvous-region logic from `exploration_manager`.

Current scope:
- subscribe to `exploration_manager/DroneState` from `/swarm_expl/drone_state`;
- use only the already-published intent prediction state;
- solve a simplified two-drone spatio-temporal rendezvous reference;
- publish RViz markers for the rendezvous center, communication-radius region, predicted positions, and diagnostic text;
- do not change exploration, task allocation, trajectory planning, or communication-risk execution.

RViz marker topic:

```text
/rendezvous_manager/two_drone_region
```

Add a `MarkerArray` display in RViz to view this topic.


## RViz display and diagnostics

This package publishes a `visualization_msgs/MarkerArray` on:

```bash
/rendezvous_manager/two_drone_region
```

Add a **MarkerArray** display in RViz and set the topic to the above name. The publisher is latched and markers have persistent lifetime, so the latest waiting/region marker remains visible after adding the display.

Quick checks:

```bash
rosnode list | grep rendezvous
rostopic hz /swarm_expl/drone_state
rostopic echo -n 1 /rendezvous_manager/two_drone_region
```

If the region cannot be generated, the node publishes an orange waiting text marker and logs the reason, for example missing drone state, stale drone state, invalid intent prediction, or low aligned intent speed.


## Communication-triggered rendezvous region generation

The rendezvous region is no longer generated continuously. The node first checks the two-drone communication state. A region is generated only when the current inter-drone distance exceeds `comm_trigger_ratio * comm_range`, or when the predicted time to reach the communication boundary is smaller than `comm_trigger_time`. This keeps the early exploration phase from being polluted by unstable rendezvous centers and allows the intent history to accumulate before the first region is created.

Key debug parameters:

```xml
<param name="enable_comm_trigger" value="true"/>
<param name="force_generate_for_debug" value="false"/>
<param name="comm_trigger_ratio" value="0.85"/>
<param name="comm_trigger_time" value="6.0"/>
<param name="region_hold_time" value="30.0"/>
<param name="min_regenerate_interval" value="5.0"/>
<param name="min_intent_integral_time" value="1.0"/>
<param name="state_timeout" value="5.0"/>
```

For pure visualization debugging, set `force_generate_for_debug=true`. For normal experiments, keep it false.


## Temporary leader/owner mode

For the current two-drone debugging stage, rendezvous-region generation is assigned to drone 1.

The node still contains the complete two-drone computation logic, but it only computes and publishes when:

```text
self_drone_id == leader_drone_id
```

Default configuration:

```xml
<arg name="self_drone_id" default="1"/>
<arg name="leader_drone_id" default="1"/>
<arg name="compute_only_if_leader" default="true"/>
```

This means drone 2 only contributes its `DroneState`; it does not publish an independent rendezvous region.
If the same node is later launched under every drone namespace, non-leader instances will stay in standby and will not publish to the rendezvous marker topic.
This avoids multiple asynchronous nodes generating inconsistent regions from different state snapshots.

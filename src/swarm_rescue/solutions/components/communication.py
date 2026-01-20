from solutions.components.state_machine import DroneState

class DroneMessage:
    class Subject:
        SWARM_INFO = "SWARM_INFO"
        MAPPING = "MAPPING"

    def __init__(self, subject: str, arg, drone_id=None):
        if subject not in vars(DroneMessage.Subject).values():
            raise ValueError(f"Invalid subject: {subject}")
        
        self.subject = subject
        self.arg = arg
        self.drone_id = drone_id

class DroneInfo:
    """Class to store drone position and metadata"""
    def __init__(self, drone_id, position, state, timestep):
        self.drone_id = drone_id
        self.position = position
        self.state = state
        self.timestep = timestep

class CommunicationManager:
    def __init__(self, drone):
        self.drone = drone
        self.wounded_locked = []
        self.other_drones_pos = []
        self.communication_params = drone.communication_params

        # Format: {drone_id: DroneInfo}
        self.latest_swarm_infos = {}

    def prepare_outgoing_messages(self):
        if self.drone.timestep_count <= 1 or self.drone.is_killed():
            return None

        messages = []

        # Share swarm informations
        self._update_own_swarm_info()
        messages.append(DroneMessage(
            subject=DroneMessage.Subject.SWARM_INFO,
            arg=self.latest_swarm_infos
        ))

        # Periodic map sharing
        if self.drone.timestep_count % self.communication_params.TIME_INTERVAL == 0:
            messages.append(DroneMessage(
                subject=DroneMessage.Subject.MAPPING,
                arg={"map": self.drone.grid.grid}
            ))

        return messages
    
    def _update_own_swarm_info(self):
        self.latest_swarm_infos[self.drone.identifier] = DroneInfo(
            drone_id=self.drone.identifier,
            position=self.drone.position,
            state=self.drone.current_state,
            timestep=self.drone.timestep_count
        )

    def process_incoming_messages(self):
        if not self.drone.communicator:
            return
            
        received_messages = self.drone.communicator.received_messages
        for msg in received_messages:
            for drone_msg in msg[1]:
                if not isinstance(drone_msg, DroneMessage):
                    raise ValueError("Invalid message type. Expected a DroneMessage instance.")
                
                self._handle_message(drone_msg)

        self._update_other_drones_pos()
        self._update_wounded_locked()

    def _info_is_recent(self, drone_info):
        """Check if the drone info is recent enough to be considered valid"""
        return (drone_info.timestep >= self.drone.timestep_count - self.communication_params.MAX_INFO_DELAY)

    def _update_other_drones_pos(self):
        self.other_drones_pos = []
        for drone_id, drone_info in self.latest_swarm_infos.items():
            if drone_id != self.drone.identifier and self._info_is_recent(drone_info):
                self.other_drones_pos.append((drone_id, drone_info.position))

    def _update_wounded_locked(self):
        self.wounded_locked = []
        wounded_lock_states = {
            DroneState.GRASPING_WOUNDED,
            DroneState.SEARCHING_RESCUE_CENTER,
            DroneState.GOING_RESCUE_CENTER
        }
        
        for drone_id, drone_info in self.latest_swarm_infos.items():
            if (drone_id != self.drone.identifier and
                drone_info.state in wounded_lock_states and
                self._info_is_recent(drone_info)):
                self.wounded_locked.append((drone_id, drone_info.position))

    def _handle_message(self, drone_msg):
        if drone_msg.subject == DroneMessage.Subject.MAPPING:
            self.drone.grid.merge_grids(drone_msg.arg["map"])

        elif drone_msg.subject == DroneMessage.Subject.SWARM_INFO:
            self._merge_swarm_info(drone_msg.arg)

    def _merge_swarm_info(self, received_data):
        """Merge received swarm info with local data, keeping most recent info
        Information can propagate between drones that aren't near each other thanks to this method."""
        for drone_id, drone_info in received_data.items():
            # If we don't have this drone's info or the received info is newer
            if drone_id not in self.latest_swarm_infos:
                self.latest_swarm_infos[drone_id] = drone_info
            elif drone_info.timestep > self.latest_swarm_infos[drone_id].timestep:
                self.latest_swarm_infos[drone_id] = drone_info
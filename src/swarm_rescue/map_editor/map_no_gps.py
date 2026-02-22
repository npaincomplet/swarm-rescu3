import math
import pathlib
import random
import sys
from typing import List, Type

# Insert the parent directory of the current file's directory into sys.path.
# This allows Python to locate modules that are one level above the current
# script, in this case spg_overlay.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
from swarm_rescue.simulation.elements.rescue_center import RescueCenter
from swarm_rescue.simulation.elements.return_area import ReturnArea
from swarm_rescue.simulation.elements.sensor_disablers import (ZoneType, NoComZone,
                                                               NoGpsZone, KillZone)
from swarm_rescue.simulation.elements.wounded_person import WoundedPerson
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.reporting.evaluation import ZonesConfig
from swarm_rescue.simulation.utils.misc_data import MiscData

from map_editor.walls_no_gps import add_walls, add_boxes, dimensions


class MyMapno_gps(MapAbstract):

    def __init__(self, drone_type: Type[DroneAbstract], zones_config: ZonesConfig = ()):
        super().__init__(drone_type=drone_type, zones_config=zones_config)
        self._max_timestep_limit = 2000
        self._max_walltime_limit = 120

        # PARAMETERS MAP
        self._size_area = dimensions()
        self._playground = ClosedPlayground(size=self._size_area)

        self._rescue_center = RescueCenter(size=(28, 9))
        self._rescue_center_pos = ((-385, 295), 0)

        self._return_area = ReturnArea(size=(35, 10))
        self._return_area_pos = ((-382, 294), 0)

        self._no_gps_zone = NoGpsZone(size=(370, 588))
        self._no_gps_zone_pos = ((200, -1), 0)

        self._wounded_persons_pos = []
        self._number_wounded_persons = len(self._wounded_persons_pos)
        self._wounded_persons: List[WoundedPerson] = []

        self._drones_pos = [(pos, random.uniform(-math.pi, math.pi)) for pos in [(-315, -213)]]
        self._number_drones = len(self._drones_pos)
        self._drones: List[DroneAbstract] = []

        # BUILD PLAYGROUND
        self._playground.add(self._return_area, self._return_area_pos)
        self._playground.add(self._rescue_center, self._rescue_center_pos)

        add_walls(self._playground)
        add_boxes(self._playground)

        self._explored_map.initialize_walls(self._playground)

        # DISABLER ZONES
        if ZoneType.NO_COM_ZONE in self._zones_config:
            self._playground.add(self._no_com_zone, self._no_com_zone_pos)

        if ZoneType.NO_GPS_ZONE in self._zones_config:
            self._playground.add(self._no_gps_zone, self._no_gps_zone_pos)

        if ZoneType.KILL_ZONE in self._zones_config:
            self._playground.add(self._kill_zone, self._kill_zone_pos)

        # POSITIONS OF THE WOUNDED PERSONS
        for i in range(self._number_wounded_persons):
            wounded_person = WoundedPerson(rescue_center=self._rescue_center)
            self._wounded_persons.append(wounded_person)
            pos = (self._wounded_persons_pos[i], 0)
            self._playground.add(wounded_person, pos)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones,
                             max_timestep_limit=self._max_timestep_limit,
                             max_walltime_limit=self._max_walltime_limit)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            self._playground.add(drone, self._drones_pos[i])

def main():
    the_map = MyMapno_gps(drone_type=DroneMotionless)
    gui = GuiSR(the_map=the_map, use_mouse_measure=True)
    gui.run()

if __name__ == '__main__':
    main()

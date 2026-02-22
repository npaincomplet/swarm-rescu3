import sys
import traceback

from swarm_rescue.simulation.reporting.evaluation import ZonesConfig
from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.map_editor.map_empty import MyMapempty
from swarm_rescue.maps.map_test_special_zones import MapTestSpecialZones
from swarm_rescue.solutions.my_drone_eval import MyDroneEval


class MyDrone(MyDroneEval):
    """Custom drone class for evaluation."""
    pass


def main():
    # Simple launcher for one map
    zones_config: ZonesConfig = (ZoneType.NO_COM_ZONE, ZoneType.NO_GPS_ZONE, ZoneType.KILL_ZONE)
    the_map = MapTestSpecialZones(drone_type=MyDrone, zones_config=zones_config)

    my_gui = GuiSR(the_map=the_map, draw_interactive=False, headless=False)

    window_title = f"Simple Launcher - Map: {type(the_map).__name__}"
    my_gui.set_caption(window_title)

    the_map.explored_map.reset()

    try:
        my_gui.run()
    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        my_gui.close()
    finally:
        # Clean up resources
        if hasattr(the_map, 'playground') and the_map.playground:
            the_map.playground.cleanup()
            try:
                the_map.playground.close_window()
            except Exception:
                pass


if __name__ == "__main__":
    main()
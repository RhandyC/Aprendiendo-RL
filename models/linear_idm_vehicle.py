import numpy as np
from highway_env.vehicle.controller import ControlledVehicle, Vehicle
from highway_env.utils import Vector
from highway_env.road.road import LaneIndex, Road, Route


import highway_env.utils as utils  # o ajusta según tu estructura


class LinearIDMVehicle(ControlledVehicle):
    """
    A vehicle using only the longitudinal IDM model (no lane change).
    """

    # IDM parameters
    ACC_MAX = 6.0  # [m/s2]
    COMFORT_ACC_MAX = 3.0  # [m/s2]
    COMFORT_ACC_MIN = -5.0  # [m/s2]
    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    TIME_WANTED = 1.5  # [s]
    DELTA_RANGE = [3.5, 4.5]  # []
    
    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: LaneIndex = None,
        target_speed: float = None,
        route: Route = None,
    ):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.target_speed = 40.0
        self.route = route

    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(
            low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1]
        )

    def act(self, action: dict | str = None):
        """
        Compute acceleration using IDM, no lateral control.
        """
        if self.crashed:
            return

        # Stay on current lane
        self.follow_road()
        action = {}
        action["steering"] = self.steering_control(self.target_lane_index)
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )

        # IDM acceleration
        front_vehicle, _ = self.road.neighbour_vehicles(self, self.lane_index)
        action["acceleration"] = self.acceleration(
            ego_vehicle=self, front_vehicle=front_vehicle
        )
        action["acceleration"] = np.clip(
            action["acceleration"], -self.ACC_MAX, self.ACC_MAX
        )

        Vehicle.act(self, action)

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
    ) -> float:
        """
        IDM longitudinal acceleration.
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(ego_target_speed, 0, ego_vehicle.lane.speed_limit)

        accel = self.COMFORT_ACC_MAX * (
            1
            - np.power(
                max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)),
                self.DELTA,
            )
        )

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            accel -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d),
                2,
            )
        return accel

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None) -> float:
        """
        Desired distance to front vehicle (standard IDM).
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = (
            np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
            if front_vehicle
            else 0
        )
        return d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))

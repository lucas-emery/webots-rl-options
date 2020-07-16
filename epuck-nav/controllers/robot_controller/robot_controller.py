from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV


class EpuckRobot(RobotEmitterReceiverCSV):
    def __init__(self):
        super().__init__(emitter_name='EPUCK_EMMITER',
                         receiver_name='EPUCK_RECEIVER',
                         timestep=32)

        self.wheel_left = self.robot.getMotor('left wheel motor')
        self.wheel_left.setPosition(float('inf'))
        self.wheel_left.setVelocity(0.0)
        self.wheel_right = self.robot.getMotor('right wheel motor')
        self.wheel_right.setPosition(float('inf'))
        self.wheel_right.setVelocity(0.0)

        self.distance_sensors = []
        for i in range(8):
            d_sensor = self.robot.getDistanceSensor('ps{}'.format(i))
            d_sensor.enable(self.get_timestep())
            self.distance_sensors.append(d_sensor)

    def create_message(self):
        message = []
        for i in range(8):
            message.append(str(self.distance_sensors[i].getValue()))
        return message

    def use_message_data(self, message):
        action = int(message[0])
        vel = self.wheel_left.getMaxVelocity() / 3

        if action == 0:     # forward
            self.wheel_left.setVelocity(vel)
            self.wheel_right.setVelocity(vel)
        elif action == 1:   # left
            self.wheel_left.setVelocity(-vel)
            self.wheel_right.setVelocity(vel)
        elif action == 2:   # right
            self.wheel_left.setVelocity(vel)
            self.wheel_right.setVelocity(-vel)
        else:
            raise Exception("Fran se la come")


# Create the robot controller object and run it
robot_controller = EpuckRobot()
robot_controller.run()

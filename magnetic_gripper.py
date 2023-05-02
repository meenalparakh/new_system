class MagneticGripper:
    def __init__(self, robot, to_ignore=[]):
        self.robot = robot
        self.robot_id = self.robot.arm.robot_id
        self.l_link = 9
        self.r_link = 10
        self.to_ignore = set(to_ignore)
        self.contact_constraint = None
        self.activated = False

    def activate(self):
        print("finding contacts")

        contact_pts_l = self.robot.pb_client.getContactPoints(
            bodyA=self.robot_id, linkIndexA=self.l_link
        )
        contact_pts_r = self.robot.pb_client.getContactPoints(
            bodyA=self.robot_id, linkIndexA=self.r_link
        )
        bodies_in_contact_l = set([c[2] for c in contact_pts_l])
        bodies_in_contact_r = set([c[2] for c in contact_pts_r])

        bodies = bodies_in_contact_l.intersection(bodies_in_contact_r).difference(
            self.to_ignore
        )

        if len(bodies) == 1:
            print("one body found in contact. Activating contact ...")
        elif len(bodies) == 0:
            print("Warning: No bodies in contact. Grip not activated.")
            return False
        else:
            print(
                f"Warning: Many bodies in contact {bodies}. Activating grasp for a random body."
            )

        body_id = list(bodies)[0]

        body_pose = self.robot.pb_client.getLinkState(self.robot_id, self.l_link)
        obj_pose = self.robot.pb_client.getBasePositionAndOrientation(body_id)
        world_to_body = self.robot.pb_client.invertTransform(body_pose[0], body_pose[1])
        obj_to_body = self.robot.pb_client.multiplyTransforms(
            world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1]
        )

        self.contact_constraint = self.robot.pb_client.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.l_link,
            childBodyUniqueId=body_id,
            childLinkIndex=-1,
            jointType=self.robot.pb_client.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=obj_to_body[0],
            parentFrameOrientation=obj_to_body[1],
            childFramePosition=(0, 0, 0),
            childFrameOrientation=(0, 0, 0),
        )

        self.activated = True

    def release(self):
        if self.activated:
            self.activated = False

        if self.contact_constraint is not None:
            self.robot.pb_client.removeConstraint(self.contact_constraint)
            self.contact_constraint = None

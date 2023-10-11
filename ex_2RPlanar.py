import nrp_manip as nrp

if __name__ == "__main__":
    num_joints = 2
    planar_rn = nrp.NRPlanar(num_joints=num_joints, rotational_axis='y')
    planar_rn.init_link_lengths(link_lengths=[1.0 for i in range(num_joints)])
    plotter = nrp.NRPlanarPlotter(robot=planar_rn)
    plotter.run()
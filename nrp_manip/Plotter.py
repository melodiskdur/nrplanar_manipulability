import matplotlib.pyplot as plt
from matplotlib import patches
import nrp_manip.NRPlanar as NRPlanar
import nrp_manip.Manipulability as Manipulability
import nrp_manip.Controller as Controller
import numpy as np


class NRPlanarPlotter:
    def __init__(self, robot: NRPlanar.NRPlanar):
        self._robot = robot
        self._fig = plt.figure("NRPlanar Plotter")
        self._ax = self._fig.subplots()
        # self._sliders = [ plt_widgets.Slider(plt.axes([0.25, 0.02 + 0.03 * i, 0.65, 0.03]), "q"+str(i+1), -np.pi, np.pi, valinit=0.05, valstep=0.1)
        #                                    for i in range(len(self._robot._joints))]
        self._v_ellipsoid: patches.Patch = None
        self._f_ellipsoid: patches.Patch = None
        self._robot_links: list = []
        self._robot_manip_data: Manipulability.ManipulabilityData
        self._text_plots: list = []

        # for slider in self._sliders:
        #    slider.on_changed(self.update_plot)

        self._ax_labels: list = []
        if self._robot._rotational_axis == 'z':
            self._ax_labels = ['x axis', 'y axis']
        elif self._robot._rotational_axis == 'y':
            self._ax_labels = ['z axis', 'x axis']
        else:
            self._ax_labels = ['y axis', 'z axis']

        self._controller_window = Controller.ControllerWindow(num_of_joints=len(self._robot._joints), update_callback=self.update_plot)

    def run(self):
        self._ax.set_xlim(1.1 * -sum(self._robot._link_lengths), 1.1 * sum(self._robot._link_lengths))
        self._ax.set_ylim(1.1 * -sum(self._robot._link_lengths), 1.1 * sum(self._robot._link_lengths))
        self.update_plot(0)
        self._ax.set_aspect('equal')
        plt.subplots_adjust(bottom=0.25)
        self._ax.set_xlabel(self._ax_labels[0])
        self._ax.set_ylabel(self._ax_labels[1])
        plt.show()

    def update_plot(self, value):
        plt.figure(self._fig)
        self._clear_plot_data()
        config = self._controller_window.config()
        v_params, f_params, eef_pos, joint_positions = self._generate_data(config=config)
        alpha = max(self._robot_manip_data.velocity_measure(), self._robot_manip_data.force_measure())
        self._v_ellipsoid = self._plot_ellipsoid(eef_pos, v_params[0], v_params[1], "Velocity Manipulability Ellipsoid", "blue", 1.0 / alpha)
        self._f_ellipsoid = self._plot_ellipsoid(eef_pos, f_params[0], f_params[1], "Force Manipulability Ellipsoid", "red", 1.0 / alpha)
        self._robot_links = self._plot_robot_links(joint_positions[0], joint_positions[1], "grey")
        self._text_plots = self._plot_annotations(v_params[0])
        self._ax.legend()
        self._fig.canvas.draw_idle()

    def _clear_plot_data(self):
        for lines in self._robot_links:
            [ line.remove() for line in lines ]
        for text in self._text_plots:
            text.remove()
        if self._v_ellipsoid != None: self._v_ellipsoid.remove()
        if self._f_ellipsoid != None: self._f_ellipsoid.remove()

    def _generate_data(self, config: list):
        fk = self._robot.forward_kinematics(config=config)
        jacobian = self._robot.numeric_jacobian(config=config)
        self._robot_manip_data = Manipulability.ManipulabilityData(jacobian)
        v_dim, v_ax = self._robot_manip_data.velocity_eigen()
        f_dim, f_ax = self._robot_manip_data.force_eigen()
        joint_positions = self._robot.calculate_joint_world_coordinates(config=config)
        axes_i = self._axes_indices()
        eef_pos = (fk[axes_i[0]], fk[axes_i[1]])
        return (v_dim, v_ax), (f_dim, f_ax), eef_pos, joint_positions

    def _axes_indices(self):
        if self._robot._rotational_axis == 'z':
            return [0 , 1]
        elif self._robot._rotational_axis == 'y':
            return [2, 0]
        else:
            return [1, 2]

    def _plot_annotations(self, eigens):
        v = "{:.3f}".format(self._robot_manip_data.velocity_measure())
        f = "{:.3f}".format(self._robot_manip_data.force_measure())
        m = "{:.3f}".format( max(eigens) / min(eigens) )
        bbox = dict(facecolor='white', edgecolor='black', boxstyle='round')
        v_text = self._fig.text(-0.50, 0.90, "Velocity Measure:\n" + v,transform=self._ax.transAxes, fontsize = 8, color='blue', bbox=bbox)
        f_text = self._fig.text(-0.50, 0.75, "Force Measure:\n" + f,transform=self._ax.transAxes, fontsize = 8, color='blue', bbox=bbox)
        m_text = self._fig.text(-0.50, 0.60, "Eigen ratio:\n" + m,transform=self._ax.transAxes, fontsize = 8, color='blue', bbox=bbox)
        return [v_text, f_text, m_text]

    def _plot_ellipsoid(self, center: tuple, v_eigen_values: list, v_eigen_vectors: list, legend: str, color: str, alpha: float):
        np_eigen = np.array(v_eigen_values)
        np_vectors = np.array([ np.array(sym_vector) for sym_vector in v_eigen_vectors])
        angle0 = np.arctan2(float(np_vectors[1, 0]), float(np_vectors[0, 0]))
        angle1 = np.arctan2(float(np_vectors[1, 1]), float(np_vectors[0, 1]))
        angle = np.degrees(min(angle0, angle1))
        ellipse_object = patches.Ellipse(center, width=alpha * np_eigen[0], height=alpha * np_eigen[1], angle=angle, fill=False, color=color, label=legend, ls=':')
        return self._ax.add_patch(ellipse_object)

    def _plot_robot_links(self, x_coords: list, y_coords: list, color: str):
        link1 = np.vstack([[0, x_coords[0]], [0, y_coords[0]]])
        links = [link1]
        for i in range(1, len(x_coords)):
            links.append( np.vstack([[links[i-1][0, 1], x_coords[i]], [links[i-1][1, 1], y_coords[i]]]) )
        ax_list = []
        for link in links:
            ax_list.append(self._ax.plot(link[0, :], link[1, :], color=color))

        return ax_list
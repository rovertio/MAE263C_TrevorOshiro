#!/usr/bin/env python3
from __future__ import annotations

# Import a plotting package ("matplotlib")
import matplotlib.pyplot as plt

# Import a multi-dimensional array and linear algebra package ("numpy")
import numpy as np

# Import the `Simulator` class from the `pydrake.systems.analysis` module in Drake
from pydrake.systems.analysis import Simulator

# Import the `DiagramBuilder` and `Diagram` classes from the `pydrake.systems.framework`
# module in Drake
from pydrake.systems.framework import DiagramBuilder, Diagram

# Import some built in `System` subclasses from the `pydrake.systems.primitives`
# module in Drake
from pydrake.systems.primitives import LogVectorOutput, Gain, Integrator, LinearSystem

# Import some `System` subclasses provided by this class to help with homework
# assignments
from mechae263C_helpers.drake import (
    LinearCombination,
    transfer_function_to_linear_system,
    plot_diagram
)

if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # TODO: Replace `...` with the correct values for each parameter
    # ----------------------------------------------------------------------------------
    Ci = 1
    Gv = 1
    Tv = 0.1/1000
    La = 2/1000
    Ra = 0.2
    ki = 1e-50
    kt = 0.2
    kv = 0.2
    Im = 0.0014
    Fm = 0.01
    Cl = 0

    # ----------------------------------------------------------------------------------
    # Add "systems" to a `DiagramBuilder` object.
    #   - "systems" are the blocks in a block diagram
    #   - Some examples for how to add named systems to a `DiagramBuilder` are given
    #     below
    # ----------------------------------------------------------------------------------
    builder = DiagramBuilder()

    # This is the summation block furthest to the left that takes the step voltage input
    # The output of a `LinearCombination` block is the linear combination of its inputs
    # with coefficients given by the `input_coeffs` argument.
    # That means for the example below if the input is u = [2, 3], the output will be
    # y = (1 * 2) + (-1 * 3) = -1
    voltage_error: LinearCombination = builder.AddNamedSystem(
        "voltage_error", LinearCombination(input_coeffs=(1, -1))
    )

    # The remaining summation blocks
    amplified_voltage: LinearCombination = builder.AddNamedSystem(
        "amplified_voltage", LinearCombination(input_coeffs=(1, -1, -1))
    )
    net_torque: LinearCombination = builder.AddNamedSystem(
        "net_torque", LinearCombination(input_coeffs=(1, -1, -1))
    )

    # Below is an example of the `Gain` system built into Drake. It takes two arguments
    #   1. k: The value of the gain
    #   2. size: The number of elements in the input value (size=1 means a scalar input)
    current_gain: Gain = builder.AddNamedSystem("current_gain", Gain(k=Ci, size=1))

    # Below is an example of the `LinearSystem` system built into Drake being created
    # using the 163C/263C helper function `transfer_function_to_linear_system`.
    # See documentation comments (by holding down the control key on and hovering your 
    # mouse the function argument names) of the provided function for more details on 
    # its input arguments.
    power_amplifier: LinearSystem = builder.AddNamedSystem(
        "power_amplifier",
        transfer_function_to_linear_system(
            continuous_time_numer_coeffs=(Gv,), continuous_time_denom_coeffs=(Tv, 1)
        ),
    )

    # TODO: Replace any `...` below with the correct system and values. Please keep the 
    # system names the same
    armature_inductance: LinearSystem = builder.AddNamedSystem(
        "armature_inductance",
        transfer_function_to_linear_system(
            continuous_time_numer_coeffs=(1,), continuous_time_denom_coeffs=(La,0)
        ),
    )
    armature_resistance: Gain = builder.AddNamedSystem(
        "armature_resistance", Gain(k=Ra, size=1)
    )
    transducer_gain: Gain = builder.AddNamedSystem(
        "transducer_gain",  Gain(k=ki, size=1)
    )
    motor_torque_constant: Gain = builder.AddNamedSystem(
        "motor_torque_constant", Gain(k=kt, size=1)
    )
    motor_inertia: LinearSystem = builder.AddNamedSystem(
        "motor_inertia",
        transfer_function_to_linear_system(
            continuous_time_denom_coeffs=(Im, 0)
        ),
    )
    motor_viscous_friction: Gain = builder.AddNamedSystem(
        "motor_viscous_friction", Gain(k=Fm, size=1)
    )
    motor_voltage_constant: Gain = builder.AddNamedSystem(
        "motor_voltage_constant", Gain(k=kv, size=1)
    )

    # This is an `Integrator` system that outputs the running integral of its input
    integrator: Integrator = builder.AddNamedSystem("integrator", Integrator(1))

    # ----------------------------------------------------------------------------------
    # Connect the systems in the `DiagramBuilder` (i.e. add arrows of block diagram)
    # ----------------------------------------------------------------------------------
    # `builder.ExportInput(input_port)` makes the provided "input_port" into an input
    # of the entire diagram
    # The functions system.get_input_port() returns the input port of the given system
    #   - If there is more than one input port, you must specify the index of the
    #     desired input
    # The functions system.get_output_port() returns the output port of the given system
    #   - If there is more than one output port, you must specify the index of the
    #     desired output
    builder.ExportInput(voltage_error.get_input_port(0))  # Export diagram input 0
    builder.ExportInput(net_torque.get_input_port(1))  # Export diagram input 1

    # The below examples show how to connect the output port of a system to the input
    # port of another system. You can think of calling this function as drawing a line
    # from one block to another in a block diagram.
    builder.Connect(voltage_error.get_output_port(), current_gain.get_input_port())
    builder.Connect(current_gain.get_output_port(), power_amplifier.get_input_port())
    builder.Connect(
        power_amplifier.get_output_port(), amplified_voltage.get_input_port(0)
    )
    builder.Connect(
        armature_resistance.get_output_port(), amplified_voltage.get_input_port(1)
    )
    builder.Connect(
        motor_voltage_constant.get_output_port(),
        amplified_voltage.get_input_port(2),
    )

    # TODO: Replace any `...` below with the correct values to connect the remaining 
    # systems.
    builder.Connect(
        amplified_voltage.get_output_port(), armature_inductance.get_input_port()
    )
    builder.Connect(
        armature_inductance.get_output_port(), transducer_gain.get_input_port()
    )
    builder.Connect(
        transducer_gain.get_output_port(), voltage_error.get_input_port(1)
    )
    builder.Connect(
        armature_inductance.get_output_port(), armature_resistance.get_input_port()
    )
    builder.Connect(
        armature_inductance.get_output_port(), motor_torque_constant.get_input_port()
    )
    builder.Connect(
        motor_torque_constant.get_output_port(), net_torque.get_input_port(0)
    )
    builder.Connect(
        motor_viscous_friction.get_output_port(), net_torque.get_input_port(2)
    )
    builder.Connect(
        net_torque.get_output_port(), motor_inertia.get_input_port()
    )
    builder.Connect(
        motor_inertia.get_output_port(), integrator.get_input_port()
    )
    builder.Connect(
        motor_inertia.get_output_port(), motor_viscous_friction.get_input_port()
    )
    builder.Connect(
        motor_inertia.get_output_port(), motor_voltage_constant.get_input_port()
    )

    # ----------------------------------------------------------------------------------
    # Log armature current and motor shaft angular velocity
    # ----------------------------------------------------------------------------------
    # These systems are special in Drake. They periodically save the output port value
    # a during a simulation so that it can be accessed later. The value is saved every
    # `publish_period` seconds in simulation time.
    #
    # Note that since we want data sampled at 1ms are publish period should be at least
    # 10 times smaller than 1ms (i.e. 1e-4).
    armature_current_logger = LogVectorOutput(
        armature_inductance.get_output_port(), builder, publish_period=1e-4
    )

    # TODO: Replace any `...` below with the correct output port to log the angular
    # velocity values
    angular_velocity_logger = LogVectorOutput(
        motor_inertia.get_output_port(), builder, publish_period=1e-4
    )

    # ----------------------------------------------------------------------------------
    # Setup/Run the simulation
    # ----------------------------------------------------------------------------------
    # This line builds a `Diagram` object and uses it to make a `Simulator` object for
    # the diagram
    diagram: Diagram = builder.Build()
    diagram.set_name("Diagram")  # This gives the Diagram a readable name
    simulator = Simulator(diagram)

    # Display Block Diagram
    plot_diagram(diagram)

    # Get the context (this contains all the information needed to run the simulation)
    context = simulator.get_mutable_context()

    # TODO: Replace any `...` below with the correct fixed value to simulate a step 
    # voltage and the specified load reaction torque input, respectively
    diagram.get_input_port(0).FixValue(context, 1)
    diagram.get_input_port(1).FixValue(context, 0)

    # Advance the simulation by 0.15 seconds using the `simulator.AdvanceTo()` function
    simulator.AdvanceTo(0.15)

    # ----------------------------------------------------------------------------------
    # Make Plots
    # ----------------------------------------------------------------------------------
    # The lines below extract the armature current log from the simulator context and
    # subsample the log to have a value every 1ms (instead of the `publish_period` of
    # 1e-4 above).
    armature_current_log = armature_current_logger.FindLog(simulator.get_context())
    ixs = np.arange(start=0, stop=len(armature_current_log.sample_times()), step=10)
    t = armature_current_log.sample_times()[ixs]
    armature_current = armature_current_log.data()[0, ixs]

    # Create a figure
    fig_motor_armature_current: plt.Figure = plt.figure()
    
    # Create an axis
    # 111 = A 1 row / 1 column grid and the 1st (and only) plot index in the grid
    ax_motor_armature_current: plt.Axes = fig_motor_armature_current.add_subplot(111)
    # Set the title of the plot
    ax_motor_armature_current.set_title("Armature Current vs Time")
    # Set the x axis label of the plot
    ax_motor_armature_current.set_xlabel("Time [s]")
    # Set the y axis label of the plot
    ax_motor_armature_current.set_ylabel("Current [A]")
    # Plot the motor armature current against time
    ax_motor_armature_current.plot(t, armature_current)

    # Save figure to the current directory as a PNG file with high resolution (dpi)
    fig_motor_armature_current.savefig("armature_current_vs_time.png", dpi=300)

    # TODO: Insert code below to plot the motor angular velocity against time and save 
    # the plot
    #...

    # This function shows the plots
    plt.show()

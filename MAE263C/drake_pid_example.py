#!/usr/bin/env python3.10

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
from pydrake.systems.primitives import (
    LogVectorOutput,
    Adder,
    Gain,
    Integrator,
    DiscreteDerivative,
    LinearSystem,
)

# Import some `System` subclasses provided by this class to help with homework
# assignments
from mechae263C_helpers.drake import (
    LinearCombination,
    transfer_function_to_linear_system,
    plot_diagram
)

if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # Set Gains
    # ----------------------------------------------------------------------------------
    Kp = 35
    Ki = 100
    Kd = 5

    natural_freq_rad_per_s = 5
    damping_ratio = 0.9

    builder = DiagramBuilder()

    # ----------------------------------------------------------------------------------
    # Make Systems
    # ----------------------------------------------------------------------------------
    error: LinearCombination = builder.AddNamedSystem(
        "error", LinearCombination(input_coeffs=(1, -1))
    )
    summation: Adder = builder.AddNamedSystem("summation", Adder(num_inputs=3, size=1))

    proportional_gain: Gain = builder.AddNamedSystem("p_gain", Gain(k=Kp, size=1))
    integral_gain: Gain = builder.AddNamedSystem("i_gain", Gain(k=Ki, size=1))
    derivative_gain: Gain = builder.AddNamedSystem("d_gain", Gain(k=Kd, size=1))

    integrator: Integrator = builder.AddNamedSystem("integral", Integrator(1))
    differentiator: Integrator = builder.AddNamedSystem(
        "derivative", DiscreteDerivative(num_inputs=1, time_step=1e-3)
    )
    plant: LinearSystem = builder.AddNamedSystem(
        "plant",
        transfer_function_to_linear_system(
            continuous_time_numer_coeffs=(natural_freq_rad_per_s ** 2,),
            continuous_time_denom_coeffs=(
                1,
                2 * damping_ratio * natural_freq_rad_per_s,
                natural_freq_rad_per_s ** 2,
            ),
        ),
    )

    # ----------------------------------------------------------------------------------
    # Export Inputs / Connect Systems
    # ----------------------------------------------------------------------------------
    builder.ExportInput(error.get_input_port(0))

    builder.Connect(error.get_output_port(), proportional_gain.get_input_port())
    builder.Connect(error.get_output_port(), integrator.get_input_port())
    builder.Connect(error.get_output_port(), differentiator.get_input_port())

    builder.Connect(integrator.get_output_port(), integral_gain.get_input_port())
    builder.Connect(differentiator.get_output_port(), derivative_gain.get_input_port())

    builder.Connect(proportional_gain.get_output_port(), summation.get_input_port(0))
    builder.Connect(integral_gain.get_output_port(), summation.get_input_port(1))
    builder.Connect(derivative_gain.get_output_port(), summation.get_input_port(2))

    builder.Connect(summation.get_output_port(), plant.get_input_port())

    builder.Connect(plant.get_output_port(), error.get_input_port(1))

    # ----------------------------------------------------------------------------------
    # Log Plant Output
    # ----------------------------------------------------------------------------------
    logger = LogVectorOutput(plant.get_output_port(), builder, publish_period=1e-3)

    # ----------------------------------------------------------------------------------
    # Setup/Run the simulation
    # ----------------------------------------------------------------------------------
    diagram: Diagram = builder.Build()
    diagram.set_name("Diagram")
    simulator = Simulator(diagram)

    # Display Block Diagram
    fig, ax = plot_diagram(diagram)
    fig.savefig("drake_pid_example_diagram.pdf")

    # Get the context (this contains all the information needed to run the simulation)
    context = simulator.get_mutable_context()

    # Fixed Step Input
    diagram.get_input_port(0).FixValue(context, 1.0)

    # Advance the simulation by 1 seconds using the `simulator.AdvanceTo()` function
    simulator.AdvanceTo(1)

    # ----------------------------------------------------------------------------------
    # Make Plots
    # ----------------------------------------------------------------------------------
    # The lines below extract the armature current log from the simulator context and
    # subsample the log to have a value every 1ms (instead of the `publish_period` of
    # 1e-4 above).
    log = logger.FindLog(simulator.get_context())
    t = log.sample_times()
    plant_output = log.data()[0, :]

    # Use the `Plot` helper class provided by this class to make a plot of the data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Plant Output vs Time for Step Input")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Plant Output")
    ax.plot(t, plant_output, label="Time Response")
    ax.axhline(1, color="black", label="Setpoint")
    ax.legend()

    # This function shows the plots
    fig.savefig("drake_pid_example_plot.pdf")
    plt.show()

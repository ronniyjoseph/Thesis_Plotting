import sys
import argparse
sys.path.append("../../Projects/hybrid_calibration/code")

from src.radiotelescope import RadioTelescope


def main():
    position_path = "../../Projects/hybrid_calibration/code/Data/MWA_Compact_Coordinates.txt"
    telescope = RadioTelescope(path=position_path)

    u_coordinates = telescope.baseline_table.u_coordinates
    v_coordinates = telescope.baseline_table.v_coordinates

    figure, axes = pyplot.subplots(1,1,figsize = (10,10))
    axes.scatter(u_coordinates, v_coordinates, s= 5)
    axes.set_title("MWA Phase II Compact")
    axes.set_xlabel("u")
    axes.set_ylabel("v")
    axes.set_aspect("equal")
    pyplot.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot
    main()
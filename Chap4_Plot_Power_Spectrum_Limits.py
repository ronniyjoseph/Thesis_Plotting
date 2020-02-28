import argparse
from scipy import interpolate
from matplotlib import colors
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy
import seaborn


def main():
    figure = pyplot.figure(figsize=(10, 10))
    axis1 = figure.add_subplot(2, 2, 1, projection='3d')
    axis2 = figure.add_subplot(2, 2, 2, projection='3d')
    axis3 = figure.add_subplot(2, 2, 3, projection='3d')
    axis4 = figure.add_subplot(2, 2, 4, projection='3d')

    plot_data3d(axis1, azimuth= -151, elevation =9)
    plot_data3d(axis2, azimuth = -179, elevation = 1, k_axis=False)
    plot_data3d(axis3, azimuth = -91, elevation = 1, redshift_axis=False)
    plot_data3d(axis4, azimuth = -90, elevation = 89, power_axis = False)

    # plot_data2d(axis2, axis1="k", axis2="power")
    # plot_data2d(axis3, axis1="z", axis2="power", y_axis=False)
    # plot_data2d(axis4, axis1="z", axis2="k")
    # handles, labels = axis3.get_legend_handles_labels()
    # figure.legend(handles, labels)
    pyplot.show()
    return


def plot_data2d(axes, axis1 = "k", axis2="power", y_axis = True):

    z_model, k_model, power_spectrum_model = load_theoretical_data()


    kolopanis19 = load_limit_data("PAPER_Limits_MKolopanis_2019.txt", z_column=0, k_column=1, limit_column=2 )

    barry19 = load_limit_data("MWA_Limits_NBarry_2019.txt", z=7, k_column=0, limit_column=1)
    li19 = load_limit_data("MWA_Limits_WLi_2019.txt", z=6.5, k_column=0, limit_column=1)
    mertens20 = load_limit_data("LOFAR_Limits_FGMertens_2020.txt", z=9.1, k_column=0, limit_column=1,
                                square_data_flag=True)
    trott20 = load_limit_data("MWA_Limits_CMTrott_2020.txt", square_data_flag=True, z_column=0, k_column=1,
                              limit_column=2 )

    cmap = pyplot.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=-4, vmax=3)

    if axis1 == "k" and axis2 == "power":
        index1 = 1
        index2 = 2
        x_label = r"$k\, [h\,\mathrm{Mpc}^{-1}]$"
        y_label = r"$\Delta^2(k)\, [\mathrm{mK}^2]$"

        x_model = numpy.log10(k_model)
        y_model = power_spectrum_model.T
        colour_plot = False

    elif axis1 == "z" and axis2 =="power":
        index1 = 0
        index2 = 2
        x_label = r"Redshift"
        y_label = r"$\Delta^2(k)\, [\mathrm{mK}^2]$"

        x_model = z_model
        y_model = power_spectrum_model
        colour_plot = False
        axes.set_xticks([6, 8, 10])



    elif axis1 == "z" and axis2 =="k":
        colour_plot = True
        index1 = 1
        index2 = 0
        x_model = numpy.log10(k_model)
        y_model = z_model
        z_model = power_spectrum_model
        x_label = r"$k\, [h\,\mathrm{Mpc}^{-1}]$"
        y_label = r"Redshift"
        axes.set_xticks([-1, 0])


    if colour_plot:
        plot = axes.pcolor(x_model, y_model, z_model, norm = norm)
        axes.scatter(kolopanis19[:, index1], kolopanis19[:, index2], s=50, marker="s", label="Kolopanis+ 2019")
        axes.scatter(li19[:, index1], li19[:, index2], s=50, marker="o", label="Li+ 2019")
        axes.scatter(barry19[:, index1], barry19[:, index2], s=50, marker="*", label="Barry+ 2019")
        axes.scatter(trott20[:, index1], trott20[:, index2], s=50, marker="v", label="Trott+ 2020")
        axes.scatter(mertens20[:, index1], mertens20[:, index2], s=50, marker="X", label="Mertens+ 2020")
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        cax = pyplot.colorbar(mappable=plot)
        cax.set_ticks([-2, 0, 2, 4, 6])
        cax.set_ticklabels([r"$10^{-2}$", r"$10^{0}$", r"$10^{2}$", r"$10^{4}$", r"$10^{6}$"])
        cax.set_label(r"$\Delta^2(k)\, [\mathrm{mK}^2]$")

    else:
        for i in range(y_model.shape[1]):
            axes.scatter(x_model, y_model[:, i], c=cmap(norm(y_model[:, i])), edgecolor='none')

        axes.scatter(kolopanis19[:, index1], kolopanis19[:, index2], s=50, marker="s", label="Kolopanis+ 2019")
        axes.scatter(li19[:, index1], li19[:, index2], s=50, marker="o", label="Li+ 2019")
        axes.scatter(barry19[:, index1], barry19[:, index2], s=50, marker="*", label="Barry+ 2019")
        axes.scatter(trott20[:, index1], trott20[:, index2], s=50, marker="v", label="Trott+ 2020")
        axes.scatter(mertens20[:, index1], mertens20[:, index2], s=50, marker="X", label="Mertens+ 2020")

        axes.set_xlabel(x_label)
        if y_axis:
            axes.set_ylabel(y_label)

    return


def plot_data3d(axes, azimuth =30, elevation = 30, redshift_axis = True, k_axis = True, power_axis = True):
    z_model, k_model, power_spectrum_model = load_theoretical_data()

    trott20 = load_limit_data("MWA_Limits_CMTrott_2020.txt", square_data_flag=True, z_column=0, k_column=1,
                              limit_column=2 )
    barry19 = load_limit_data("MWA_Limits_NBarry_2019.txt", z=7, k_column=0, limit_column=1)
    li19 = load_limit_data("MWA_Limits_WLi_2019.txt", z=6.5, k_column=0, limit_column=1)
    mertens20 = load_limit_data("LOFAR_Limits_FGMertens_2020.txt", z=9.1, k_column=0, limit_column=1,
                                square_data_flag=True)
    kolopanis19 = load_limit_data("PAPER_Limits_MKolopanis_2019.txt", z_column=0, k_column=1, limit_column=2 )

    cmap = pyplot.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=-4, vmax=3)

    zz, kk = numpy.meshgrid(z_model, numpy.log10(k_model))
    axes.plot_surface(kk, zz, power_spectrum_model.T, cmap=cmap,norm=norm, antialiased = False)

    axes.scatter(kolopanis19[:, 1], kolopanis19[:, 0], kolopanis19[:, 2], s=50, marker="s", label="Kolopanis+ 2019")
    axes.scatter(li19[:, 1], li19[:, 0], li19[:, 2], s=50, marker = "o", label = "Li+ 2019")
    axes.scatter(barry19[:, 1], barry19[:, 0], barry19[:, 2], s=50, marker="*", label="Barry+ 2019")
    axes.scatter(trott20[:, 1], trott20[:, 0], trott20[:, 2], s=50, marker="v", label="Trott+ 2020")
    axes.scatter(mertens20[:, 1], mertens20[:, 0], mertens20[:, 2], s=50, marker="X", label="Mertens+ 2020")

    if redshift_axis:
        axes.set_yticks([6, 8, 10])
        axes.set_ylabel(r"Redshift", labelpad=8)
    else:
        axes.yaxis.set_ticklabels([])

    if k_axis:
        axes.set_xticks([-1, 0])
        axes.set_xticklabels([r"$10^{-1}$", r"$10^{0}$"])
        axes.set_xlabel(r"$k\, [h\,\mathrm{Mpc}^{-1}]$",labelpad=8)
    else:
        axes.xaxis.set_ticklabels([])


    if power_axis:
        axes.set_zticks([-2, 0, 2, 4, 6])
        axes.set_zticklabels([r"$10^{-2}$", r"$10^{0}$", r"$10^{2}$", r"$10^{4}$", r"$10^{6}$"])
        axes.set_zlabel(r"$\Delta^2(k)\, [\mathrm{mK}^2]$", labelpad=8)
    else:
        axes.zaxis.set_ticklabels([])

    minor_ticks_log = numpy.arange(0, 10, 1)
    minimum_order = numpy.log10(k_model.min())/numpy.abs(numpy.log10(k_model.min()))*numpy.ceil(numpy.abs(numpy.log10((k_model.min()))))
    maximum_order = numpy.log10(k_model.max())/numpy.abs(numpy.log10(k_model.max()))*numpy.ceil(numpy.abs(numpy.log10((k_model.max()))))
    orders = numpy.arange(minimum_order, maximum_order, 1)

    axes.view_init(azim =azimuth, elev=elevation)

    minor_tick_positions = []
    for i in range(len(orders)):
        steps = (1 + minor_ticks_log)*10**orders[i]
        lower_index = numpy.where(steps > k_model.min())[0]
        upper_index = numpy.where(steps < k_model.max())[0]
        if i == 0:
            minor_tick_positions = steps[lower_index[0]:upper_index[-1]+1]
        else:
            minor_tick_positions = numpy.append(minor_tick_positions, steps[lower_index[0]:upper_index[-1]+1])
    return


def load_limit_data(file_name, limit_folder= "Observed_PS/", z = None, z_column = None, k_column = 0, limit_column = 1,
                    square_data_flag = False):
    data = numpy.loadtxt(limit_folder + file_name)
    limit_data = numpy.zeros((data.shape[0], 3))
    if z is None:
        limit_data[:, 0] = data[:, z_column]
    if z is not None:
        limit_data[:, 0] = numpy.zeros(data.shape[0]) + z
    limit_data[:, 1] = numpy.log10(data[:, k_column])
    if square_data_flag:
        limit_data[:, 2] = numpy.log10(data[:, limit_column]**2)
    else:
        limit_data[:, 2] = numpy.log10(data[:, limit_column])

    return limit_data


def load_theoretical_data(model_folder="Mesinger_PS_Faint_Galaxies/", log_k_min = -1.5, log_k_max = None):

    filename_break1 = "ps_no_halos_z"
    filename_break2 = "_nf0"

    filename_list = os.listdir(model_folder)

    filename_split = [datafile.split(filename_break1, 1)[1] for datafile in filename_list]
    redshift_list = [datafile.split(filename_break2, 1)[0] for datafile in filename_split]

    redshifts = numpy.array(list(map(float, redshift_list)))
    sorted_redshifts = redshifts[redshifts.argsort()]

    #Open the first file to figure out the dimensions of the data
    first_file = numpy.loadtxt(model_folder + filename_list[0])
    k_data = first_file[:, 0]

    #Define interpolation range
    if log_k_max is None:
        log_k_max = numpy.log10(k_data[-1])
    k_bins = numpy.logspace(log_k_min, log_k_max, 50)

    power_spectrum_model = numpy.zeros((len(redshifts), len(k_bins)))
    for i in range(len(sorted_redshifts)):
        file_index = numpy.where(sorted_redshifts[i] == redshifts)[0][0]
        data = numpy.loadtxt(model_folder + filename_list[file_index])
        interpolation = interpolate.interp1d(numpy.log10(data[:, 0]), numpy.log10(data[:, 1]))
        power_spectrum_model[i, :] = interpolation(numpy.log10(k_bins))

    return sorted_redshifts, k_bins, power_spectrum_model


def colorbar(mappable, extend='neither'):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, extend = extend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot
    main()

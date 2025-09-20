{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/mohammadilz12/Assignment1/blob/main/Assignment_02_Python_Packages_mohammadilz12.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 📚 Assignment 2 — NumPy, pandas & Matplotlib Essentials\n",
        "<p align=\"center\">📢⚠️📂  </p>\n",
        "\n",
        "<p align=\"center\"> Please name your file using the format: <code>assignmentName_nickname.py/.ipynb</code> (e.g., <code>project1_ali.py</code>) and push it to GitHub with a clear commit message.</p>\n",
        "\n",
        "<p align=\"center\"> 🚨📝🧠 </p>\n",
        "\n",
        "\n",
        "------------------------------------------------\n",
        "\n",
        "\n",
        "Welcome to your first data-science sprint! In this three-part mini-project you’ll touch the libraries every ML practitioner leans on daily:\n",
        "\n",
        "1. NumPy — fast n-dimensional arrays\n",
        "\n",
        "2. pandas — tabular data wrangling\n",
        "\n",
        "3. Matplotlib — quick, customizable plots\n",
        "\n",
        "Each part starts with “Quick-start notes”, then gives you two bite-sized tasks. Replace every # TODO with working Python in a notebook or script, run it, and check your results. Happy hacking! 😊\n"
      ],
      "metadata": {
        "id": "ZS5lje_18xC2"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 1 · NumPy 🧮\n",
        "Quick-start notes\n",
        "Core object: ndarray (n-dimensional array)\n",
        "\n",
        "Create data: np.array, np.arange, np.random.*\n",
        "\n",
        "Summaries: mean, std, sum, max, …\n",
        "\n",
        "Vectorised math beats Python loops for speed\n",
        "\n"
      ],
      "metadata": {
        "id": "u5vvtK-6840I"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Task 1 — Mock temperatures\n",
        "\n"
      ],
      "metadata": {
        "id": "UyNHtkGm9OgH"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lmaiboJe8E15"
      },
      "outputs": [],
      "source": [
        "# 👉 # TODO: import numpy and create an array of 365\n",
        "# normally-distributed °C values (µ=20, σ=5) called temps\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Task 2 — Average temperature\n"
      ],
      "metadata": {
        "id": "BtsB8QKM9Xs_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 👉 # TODO: print the mean of temps\n"
      ],
      "metadata": {
        "id": "DSR8aS-F9Z10"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 2 · pandas 📊\n",
        "Quick-start notes\n",
        "Main structures: DataFrame, Series\n",
        "\n",
        "Read data: pd.read_csv, pd.read_excel, …\n",
        "\n",
        "Selection: .loc[label], .iloc[pos]\n",
        "\n",
        "Group & summarise: .groupby(...).agg(...)\n",
        "\n"
      ],
      "metadata": {
        "id": "8JjHX4wy9dPz"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Task 3 — Load ride log\n"
      ],
      "metadata": {
        "id": "wRyJyhbt9uUr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 👉 # TODO: read \"rides.csv\" into df\n",
        "# (columns: date,temp,rides,weekday)\n"
      ],
      "metadata": {
        "id": "1J4jcLct9yVO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Task 4 — Weekday averages"
      ],
      "metadata": {
        "id": "byKd8SFK9w2y"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 👉 # TODO: compute and print mean rides per weekday\n"
      ],
      "metadata": {
        "id": "4rLrxkPj90p3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 3 · Matplotlib 📈\n",
        "Quick-start notes\n",
        "Workhorse: pyplot interface (import matplotlib.pyplot as plt)\n",
        "\n",
        "Figure & axes: fig, ax = plt.subplots()\n",
        "\n",
        "Common plots: plot, scatter, hist, imshow\n",
        "\n",
        "Display inline in Jupyter with %matplotlib inline or %matplotlib notebook"
      ],
      "metadata": {
        "id": "gbGMbtLf94M4"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Task 5 — Scatter plot"
      ],
      "metadata": {
        "id": "QOsToPh2-AnZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 👉 # TODO: scatter-plot temperature (x) vs rides (y) from df\n",
        "\n"
      ],
      "metadata": {
        "id": "_pCU2mIH-DAi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Task 6 — Show the figure\n",
        "\n"
      ],
      "metadata": {
        "id": "TBGXIzVV-E5u"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 👉 # TODO: call plt.show() so the plot appears\n",
        "\n"
      ],
      "metadata": {
        "id": "ycvsNyqh-GRM"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}